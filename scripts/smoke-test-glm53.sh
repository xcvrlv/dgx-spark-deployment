#!/usr/bin/env bash
set -euo pipefail

base_url="${BASE_URL:-http://127.0.0.1:8000/v1}"
model="${SERVED_MODEL_NAME:-glm-5.3-flash}"
expected_max_model_len="${EXPECTED_MAX_MODEL_LEN:-}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT

post_json() {
  local payload="$1" output="$2"
  curl --silent --show-error --fail --max-time 300 \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    "$base_url/chat/completions" >"$output"
}

curl --silent --show-error --fail --max-time 10 "$base_url/models" >"$tmp_dir/models.json"
python3 - "$tmp_dir/models.json" "$model" "$expected_max_model_len" <<'PY'
import json
import sys

body = json.load(open(sys.argv[1], encoding="utf-8"))
available = {item["id"] for item in body.get("data", [])}
assert sys.argv[2] in available, (sys.argv[2], sorted(available))
if sys.argv[3]:
    card = next(item for item in body["data"] if item["id"] == sys.argv[2])
    reported = card.get("max_model_len")
    if reported is not None:
        assert int(reported) == int(sys.argv[3]), (reported, sys.argv[3])
PY

post_json "$(cat <<JSON
{"model":"$model","messages":[{"role":"user","content":"Reply with a short greeting and state your model family."}],"temperature":0,"max_tokens":128,"logprobs":true,"top_logprobs":1,"chat_template_kwargs":{"reasoning_effort":"low"}}
JSON
)" "$tmp_dir/basic.json"
python3 - "$tmp_dir/basic.json" <<'PY'
import json
import math
import sys

body = json.load(open(sys.argv[1], encoding="utf-8"))
assert not body.get("error"), body
choice = body["choices"][0]
message = choice["message"]
visible = (message.get("content") or "") + (message.get("reasoning_content") or "")
assert visible.strip(), body
tokens = (choice.get("logprobs") or {}).get("content") or []
assert tokens, "no token logprobs returned"
assert all(math.isfinite(token["logprob"]) for token in tokens), "non-finite token logprob"
PY

post_json "$(cat <<JSON
{"model":"$model","messages":[{"role":"user","content":"Call echo_word with the word spark."}],"temperature":0,"max_tokens":256,"tool_choice":"required","tools":[{"type":"function","function":{"name":"echo_word","description":"Echo one word","parameters":{"type":"object","properties":{"word":{"type":"string"}},"required":["word"]}}}],"chat_template_kwargs":{"reasoning_effort":"low"}}
JSON
)" "$tmp_dir/tool.json"
python3 - "$tmp_dir/tool.json" <<'PY'
import json
import sys

body = json.load(open(sys.argv[1], encoding="utf-8"))
assert not body.get("error"), body
calls = body["choices"][0]["message"].get("tool_calls") or []
assert calls, body
assert calls[0]["function"]["name"] == "echo_word", calls
arguments = calls[0]["function"]["arguments"]
if isinstance(arguments, str):
    arguments = json.loads(arguments)
assert arguments.get("word", "").lower() == "spark", arguments
PY

echo "GLM-5.3 smoke test passed: model listing, finite decode, and tool calling"
