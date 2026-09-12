#!/usr/bin/env python3
"""N concurrent distinct needle prompts of ~<tokens> each. Prints per-request result and wall time."""
import json, urllib.request, time, sys, random, concurrent.futures as cf
URL = "http://10.0.0.1:8888/v1/chat/completions"
target = int(sys.argv[1]); n = int(sys.argv[2]); base = int(sys.argv[3]) if len(sys.argv) > 3 else 100
words = ["harbour","lantern","meadow","copper","violin","orchard","compass","thistle","ember","granite","willow","saffron","anchor","quartz","ledger","falcon"]
def build(seed, reps):
    rnd = random.Random(seed)
    para = lambda i: f"Entry {seed}-{i}: the {rnd.choice(words)} keeper noted {rnd.randint(1,999)} {rnd.choice(words)} crates by the {rnd.choice(words)} gate before the {rnd.choice(words)} bell rang at dusk. "
    return "".join(para(i) for i in range(reps)) + f"\nThe secret code word is PELICAN-{seed}.\n" + "".join(para(reps + i) for i in range(20)) + "\nWhat is the secret code word? Answer with the code word only."
def ask(p, n=20):
    body = dict(model="deepseek-v4.1-flash", temperature=0, max_tokens=n, stream=False, chat_template_kwargs={"thinking": False}, messages=[{"role": "user", "content": p}])
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3000))
    return r["usage"]["prompt_tokens"], time.time() - t0, r["choices"][0]["message"]["content"].strip()
n50, _, _ = ask(build(1, 50), 1); per = (n50 - 20) / 50.0; reps = int((target - 200) / per)
t0 = time.time()
with cf.ThreadPoolExecutor(n) as ex:
    rs = list(ex.map(lambda s: ask(build(s, reps)), range(base, base + n)))
wall = time.time() - t0
for i, (pt, dt, txt) in enumerate(rs):
    print(f"  [{i}] prompt_tokens={pt} {dt:.1f}s answer={txt[:20]!r} {'OK' if txt == f'PELICAN-{base+i}' else 'WRONG'}")
print(f"CONC {n} x ~{target}: wall {wall:.1f}s, total prompt tokens {sum(r[0] for r in rs)}")
