"""C5: memory write gate with evidence and signed instructions (arm Wqs, shard m1s, p4m-1.4, 4B agent), paired by case_id with Wq (m1q, write gate + evidence)
and Emem (m1, rules + gate) and W (m1, write gate). Writes paper/numbers_c5_<agent>.json. Usage: python -I paper/scripts/c5_analysis.py [Qwen3-4B]"""
import json, os, sys
from math import comb

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RAW = os.path.join(ROOT, "pilot_runs", "raw")
agent = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-4B"


def load(n):
    return [json.loads(l) for l in open(os.path.join(RAW, n), encoding="utf-8")]


def mcnemar(x, y):
    n = x + y
    if n == 0:
        return 1.0
    k = min(x, y)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


SRC = {"Wqs": f"m1s_{agent}_r0_events.jsonl", "Wq": f"m1q_{agent}_r0_events.jsonl", "Emem": f"m1_{agent}_r0_events.jsonl", "W": f"m1_{agent}_r0_events.jsonl"}
O = {}
for arm, f in SRC.items():
    for r in load(f):
        if r["kind"] == "outcome" and r["arm"] == arm:
            O[(arm, r["group"], r["case_id"])] = r
safe = lambda r: int(r["utility"] and not r["attack_success"])
M = dict(asr=lambda r: int(r["attack_success"]), done=lambda r: int(r["utility"]), safe=safe)
att = sorted({k[2] for k in O if k[0] == "Wqs" and k[1] == "attack"})
out = dict(agent=agent, n_attacked=len(att), arms={}, pairs={})
for arm in ("W", "Wq", "Emem", "Wqs"):
    a = [O[(arm, "attack", c)] for c in att if (arm, "attack", c) in O]
    hb = [r for k, r in O.items() if k[0] == arm and k[1] == "hard_benign"]
    be = [r for k, r in O.items() if k[0] == arm and k[1] == "benign"]
    out["arms"][arm] = dict(attacked=len(a), **{m: sum(f(r) for r in a) for m, f in M.items()},
                            sup_calls=sum(r.get("sup_calls", 0) for r in a), sup_blocks=sum(r.get("sup_blocks", 0) for r in a),
                            sup_overruled=sum(r.get("sup_overruled", 0) for r in a), sig_rejects=sum(r.get("sig_rejects", 0) for r in a),
                            hard_benign=[sum(r["utility"] for r in hb), len(hb)], benign=[sum(r["utility"] for r in be), len(be)])
for other in ("Wq", "Emem", "W"):
    for m, f in M.items():
        ks = [c for c in att if ("Wqs", "attack", c) in O and (other, "attack", c) in O]
        x = sum(1 for c in ks if f(O[("Wqs", "attack", c)]) and not f(O[(other, "attack", c)]))
        y = sum(1 for c in ks if f(O[(other, "attack", c)]) and not f(O[("Wqs", "attack", c)]))
        out["pairs"][f"Wqs_vs_{other}_{m}"] = dict(n=len(ks), only_Wqs=x, only_other=y, p=round(mcnemar(x, y), 4))
json.dump(out, open(os.path.join(ROOT, "paper", f"numbers_c5_{agent}.json"), "w"), indent=1)
for k, v in out["arms"].items():
    print(k, v)
print(json.dumps(out["pairs"]))
