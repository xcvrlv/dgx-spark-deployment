#!/usr/bin/env python3
"""Step 1: greedy batches of 3 and 4 (prose prompts, 120 tokens) + batch of 4 at temperature 0.7.
Prints first 80 chars of each response and a garbage verdict."""
import json, time, sys, urllib.request, concurrent.futures as cf, re, statistics
URL = "http://10.0.0.1:8888/v1/chat/completions"
PROMPTS = [
 "Write a vivid short story (about 500 words) about a lighthouse keeper's last winter. Prose only, no headings.",
 "Write a vivid short story (about 500 words) about a city that only exists at night. Prose only, no headings.",
 "Write a vivid short story (about 500 words) about two rivals sharing a train compartment. Prose only, no headings.",
 "Write a vivid short story (about 500 words) about a violin found in a flooded cellar. Prose only, no headings.",
 "Describe, in flowing prose, a morning walk through a market town in early autumn.",
 "Explain in plain English how a bicycle's gears change the effort needed to climb a hill.",
]
def gen(i, temp, n=120):
    body = dict(model="deepseek-v4.1-flash", temperature=temp, max_tokens=n, stream=False,
                chat_template_kwargs={"thinking": False},
                messages=[{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}])
    if temp > 0: body["top_p"] = 0.95
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t0
    c = r["choices"][0]
    return r["usage"]["completion_tokens"], dt, c["finish_reason"], c["message"]["content"]
def garbage(txt):
    # Latin prose expected: flag if <70% of non-space chars are ASCII letters/punct, or CJK present
    if not txt.strip(): return True
    cjk = re.search(r'[　-鿿가-힯]', txt) is not None
    body = [ch for ch in txt if not ch.isspace()]
    latin = sum(1 for ch in body if ch.isascii() and (ch.isalpha() or ch in ".,;:'\"!?-()"))
    return cjk or (latin / max(1, len(body)) < 0.7)
def level(c, temp, k=0, n=120):
    t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex:
        rs = list(ex.map(lambda i: gen(i, temp, n), range(k, k + c)))
    wall = time.time() - t0
    tot = sum(r[0] for r in rs)
    bad = 0
    print(f"batch={c} temp={temp}: aggregate {tot/wall:5.1f} tok/s wall {wall:5.1f}s tokens {tot}")
    for i, (nt, dt, fr, txt) in enumerate(rs):
        g = garbage(txt); bad += g
        print(f"   [{i}] {nt:4d} tok {nt/dt:5.1f} tok/s {'GARBAGE' if g else 'ok     '} {txt[:80]!r}")
    print(f"   => {bad}/{c} garbled")
    return bad
if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    print(f"== batch test {tag} {time.strftime('%H:%M:%S')} ==")
    total = 0
    total += level(3, 0.0)
    total += level(4, 0.0)
    total += level(4, 0.7)
    print(f"RESULT {tag}: {total} garbled responses total")
