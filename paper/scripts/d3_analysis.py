"""D3 (V62): Qwen3-8B (4-bit) as supervisor for the Qwen3-4B agent, arms Bq/Eq, 222 cases, paired by case_id with the Qwen3-4B supervisor
(shard s3q, same agent, same cases). Writes paper/numbers_d3.json. Usage: python -I paper/scripts/d3_analysis.py"""
import collections, json, os
from math import comb

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RAW = os.path.join(ROOT, "pilot_runs", "raw")


def load(name):
    return [json.loads(l) for l in open(os.path.join(RAW, name), encoding="utf-8")]


def mcnemar(x, y):
    n = x + y
    if n == 0:
        return 1.0
    k = min(x, y)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


d3 = load("d3_Qwen3-4B_r0_sup-Qwen3-8B_v62_events.jsonl")
s3 = load("s3q_Qwen3-4B_r0_events.jsonl")
meta = [r for r in d3 if r["kind"] == "run_meta"][0]
oc = lambda rows, arm: {r["case_id"]: r for r in rows if r["kind"] == "outcome" and r["arm"] == arm}
out = dict(harness_d3=meta["harness"], sup_model=meta["sup_model"], sup_quant=meta.get("sup_quant"), harness_s3q=sorted({r.get("harness") for r in s3 if r["kind"] == "run_meta"}),
           kinds=dict(collections.Counter(r["kind"] for r in d3)), arms={}, pairs={})
metrics = dict(asr=lambda r: int(r["attack_success"]), done=lambda r: int(r["utility"]), safe=lambda r: int(r["utility"] and not r["attack_success"]))
O = {}
for tag, rows in (("sup4b", s3), ("sup8b", d3)):
    for arm in ("Bq", "Eq"):
        O[(tag, arm)] = oc(rows, arm)
for (tag, arm), d in O.items():
    for grp in sorted({r["group"] for r in d.values()}):
        rs = [r for r in d.values() if r["group"] == grp]
        out["arms"][f"{tag}:{arm}:{grp}"] = dict(n=len(rs), **{m: sum(f(r) for r in rs) for m, f in metrics.items()},
                                                   sup_blocks=sum(r.get("sup_blocks", 0) for r in rs), parse_fail=sum(r.get("sup_parse_fail", 0) for r in rs))
for arm in ("Bq", "Eq"):
    a, b = O[("sup4b", arm)], O[("sup8b", arm)]
    for grp in sorted({r["group"] for r in b.values()}):
        cs = [c for c in b if c in a and b[c]["group"] == grp]
        for m, f in metrics.items():
            x = sum(1 for c in cs if f(a[c]) and not f(b[c]))
            y = sum(1 for c in cs if f(b[c]) and not f(a[c]))
            out["pairs"][f"{arm}:{grp}:{m}"] = dict(n=len(cs), only_4b_sup=x, only_8b_sup=y, p=round(mcnemar(x, y), 4))
json.dump(out, open(os.path.join(ROOT, "paper", "numbers_d3.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
