"""C1: hybrid supervisor with block feedback (Ef) for the 1.7B agent (shard c1), paired by case_id with E (s2, hybrid without feedback),
Eq (s3q, hybrid + evidence) and Bf (s3q, independent + feedback), same seeds. Writes paper/numbers_c1_<agent>.json.
Usage: python -I paper/scripts/c1_analysis.py [Qwen3-1.7B]"""
import json, os, sys
from math import comb

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RAW = os.path.join(ROOT, "pilot_runs", "raw")
agent = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-1.7B"


def load(n):
    return [json.loads(l) for l in open(os.path.join(RAW, n), encoding="utf-8")]


def mcnemar(x, y):
    n = x + y
    if n == 0:
        return 1.0
    k = min(x, y)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


SRC = {"Ef": f"c1_{agent}_r0_events.jsonl", "E": f"s2_{agent}_r0_events.jsonl", "Eq": f"s3q_{agent}_r0_events.jsonl", "Bf": f"s3q_{agent}_r0_events.jsonl"}
O = {}
for arm, f in SRC.items():
    for r in load(f):
        if r["kind"] == "outcome" and r["arm"] == arm:
            O[(arm, r["group"], r["case_id"])] = r
safe = lambda r: int(r["utility"] and not r["attack_success"])
M = dict(asr=lambda r: int(r["attack_success"]), done=lambda r: int(r["utility"]), safe=safe)
att = sorted({k[2] for k in O if k[0] == "Ef" and k[1] == "attack"})
out = dict(agent=agent, n_attacked=len(att), arms={}, pairs={})
for arm in ("E", "Eq", "Bf", "Ef"):
    a = [O[(arm, "attack", c)] for c in att if (arm, "attack", c) in O]
    hb = [r for k, r in O.items() if k[0] == arm and k[1] == "hard_benign"]
    be = [r for k, r in O.items() if k[0] == arm and k[1] == "benign"]
    out["arms"][arm] = dict(attacked=len(a), **{m: sum(f(r) for r in a) for m, f in M.items()}, sup_calls=sum(r["sup_calls"] for r in a),
                            sup_blocks=sum(r["sup_blocks"] for r in a), steps_after_block=sum(r["steps_after_block"] for r in a),
                            hard_benign=[sum(r["utility"] for r in hb), len(hb)], benign=[sum(r["utility"] for r in be), len(be)])
for other in ("E", "Eq", "Bf"):
    for m, f in M.items():
        ks = [c for c in att if ("Ef", "attack", c) in O and (other, "attack", c) in O]
        x = sum(1 for c in ks if f(O[("Ef", "attack", c)]) and not f(O[(other, "attack", c)]))
        y = sum(1 for c in ks if f(O[(other, "attack", c)]) and not f(O[("Ef", "attack", c)]))
        out["pairs"][f"Ef_vs_{other}_{m}"] = dict(n=len(ks), only_Ef=x, only_other=y, p=round(mcnemar(x, y), 4))
json.dump(out, open(os.path.join(ROOT, "paper", f"numbers_c1_{agent}.json"), "w"), indent=1)
print(json.dumps(out["arms"])); print(json.dumps(out["pairs"]))
