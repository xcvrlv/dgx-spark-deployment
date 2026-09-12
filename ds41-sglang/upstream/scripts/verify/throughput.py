#!/usr/bin/env python3
"""Greedy and sampled throughput at concurrency 1-4 (prose prompts, 300 tokens)."""
import json, time, urllib.request, concurrent.futures as cf, statistics, sys
URL = "http://10.0.0.1:8888/v1/chat/completions"
TOPICS = ["a lighthouse keeper's last winter", "a city that only exists at night", "two rivals sharing a train compartment",
 "a violin found in a flooded cellar", "the first market day after a long war", "a cartographer who maps dreams",
 "a bakery run by retired sailors", "an orchard planted on a rooftop", "a letter delivered forty years late",
 "a clockmaker's apprentice in a silent town", "a storm seen from a mountain hut", "a chess club in a fishing village"]
def gen(i, temp, n=300):
    body = dict(model="deepseek-v4.1-flash", temperature=temp, max_tokens=n, stream=False, chat_template_kwargs={"thinking": False},
                messages=[{"role": "user", "content": f"Write a vivid short story (about 500 words) about {TOPICS[i % len(TOPICS)]}. Prose only, no headings."}])
    if temp > 0: body["top_p"] = 0.95
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t0
    c = r["choices"][0]
    return r["usage"]["completion_tokens"], dt, c["finish_reason"], c["message"]["content"][:50].replace("\n", " ")
def level(c, temp, k):
    t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex: rs = list(ex.map(lambda i: gen(i, temp), range(k, k + c)))
    wall = time.time() - t0; per = [n / dt for n, dt, _, _ in rs]; tot = sum(n for n, *_ in rs)
    print(f"conc={c} temp={temp}: per-request {statistics.mean(per):5.1f} tok/s (min {min(per):5.1f}), aggregate {tot/wall:5.1f} tok/s, wall {wall:5.1f}s, tokens {tot}", flush=True)
    for n, dt, fr, txt in rs: print(f"      {n:4d} tok in {dt:5.1f}s finish={fr:6s} {txt!r}")
    return tot / wall
res = {}
for temp in (0.0, 0.7):
    for c in (1, 2, 3, 4):
        res[f"conc{c}_t{temp}"] = round(level(c, temp, c), 1)
print("SUMMARY", json.dumps(res))
