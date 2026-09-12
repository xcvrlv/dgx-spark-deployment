#!/usr/bin/env python3
"""Boot-3 probe sequence (no warm-up ran): is the prefill path clean on a virgin engine?
Then replay the warm-up pieces one at a time with a probe after each."""
import json, time, sys, urllib.request, concurrent.futures as cf, math, random
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from batchtest import PROMPTS, garbage
URL = "http://10.0.0.1:8888/v1/chat/completions"
core = " Now write a vivid short story about a lighthouse keeper's last winter. Prose only."
def prompt_with(n_fill, seed):
    rnd = random.Random(seed)
    return (f"Record {seed}: " + " ".join(str(rnd.randint(0, 9)) for _ in range(n_fill)) + "." + core) if n_fill else core.strip()
def gen(prompt, n=40, temp=0.0, logprobs=True):
    body = dict(model="deepseek-v4.1-flash", temperature=temp, max_tokens=n, stream=False,
                chat_template_kwargs={"thinking": False}, messages=[{"role": "user", "content": prompt}])
    if logprobs: body["logprobs"] = True; body["top_logprobs"] = 1
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900))
    c = r["choices"][0]; lp = [t["logprob"] for t in c["logprobs"]["content"]] if logprobs else [None]
    return r["usage"]["prompt_tokens"], c["message"]["content"], lp
def probe(tag, n_fill=280, seed=None):
    seed = seed or random.randint(1, 10**6)
    pt, txt, lp = gen(prompt_with(n_fill, seed))
    bad = garbage(txt) or lp[0] is None
    print(f"PROBE {tag}: prompt_tokens={pt} lp0={'NaN' if lp[0] is None else round(lp[0],3)} {'GARBAGE' if bad else 'ok'} {txt[:60]!r}", flush=True)
    return bad
def batch(tag, k, n=40):
    with cf.ThreadPoolExecutor(k) as ex:
        rs = list(ex.map(lambda i: gen(PROMPTS[i % 6], n), range(k)))
    bad = sum(1 for _, t, lp in rs if garbage(t) or lp[0] is None)
    print(f"BATCH {tag} (bs={k}): {bad}/{k} garbled  " + " | ".join(t[:25].replace(chr(10),' ') for _, t, _ in rs), flush=True)
    return bad
print(f"== diag5 {time.strftime('%H:%M:%S')} virgin engine (WARMUP=0) ==")
b = probe("virgin 600-token prompt")
b += probe("virgin 150-token prompt", n_fill=55)
b += batch("virgin", 3)
b += batch("virgin", 4)
if b:
    print("RESULT: garbage on a virgin engine -> the warm-up is not the trigger; configuration bisect next"); sys.exit(0)
print("== clean so far: replay the warm-up pieces, probe after each ==")
filler = ('The quick brown fox jumps over the lazy dog near the riverbank while the sun sets slowly behind the distant hills. ')
for words in (12, 200, 900, 3600):
    p = (filler * (words // 20 + 1)) + 'Summarise the text above in one sentence.'
    pt, txt, lp = gen(p, 24, logprobs=False)
    print(f"warm-up piece ~{words} words: prompt_tokens={pt} {txt[:50]!r}", flush=True)
    if probe(f"after warm-up {words} words") or batch(f"after warm-up {words} words", 3):
        print(f"RESULT: state broke after the ~{words}-word warm-up prompt"); sys.exit(0)
wb = ['Explain how a hash table handles collisions in two sentences.', 'List the steps of the TCP three-way handshake.',
      'What is the capital of France? Answer in one word.', 'Write a haiku about mountains.']
with cf.ThreadPoolExecutor(4) as ex: list(ex.map(lambda p: gen(p, 48, logprobs=False), wb))
if probe("after warm-up batch of 4") or batch("after warm-up batch of 4", 3):
    print("RESULT: state broke after the warm-up batch of 4"); sys.exit(0)
print("== still clean: longer soak: sweep 100..9000 tokens and repeated batches ==")
for k, target in enumerate((100, 300, 900, 1700, 2300, 3000, 4500, 9000)):
    b += probe(f"sweep {target}", n_fill=max(0, (target - 45) // 2), seed=3000 + k)
for rep in range(3):
    b += batch(f"soak rep{rep}", 3) + batch(f"soak rep{rep}", 4)
print(f"RESULT: {b} garbled after the full replay; " + ("CLEAN engine" if b == 0 else "garbage appeared late"))
