"""G9 tolerant evidence match: Bt/Et (shard g9t, p4h-3.7) paired by case_id with Bq/Eq (s3q, exact-after-normalisation match), same agent, supervisor and seeds.
Writes paper/numbers_g9t_<agent>.json. Usage: python -I paper/scripts/g9t_analysis.py [Qwen3-4B]"""
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


SRC = {"Bt": f"g9t_{agent}_r0_events.jsonl", "Et": f"g9t_{agent}_r0_events.jsonl", "Bq": f"s3q_{agent}_r0_events.jsonl", "Eq": f"s3q_{agent}_r0_events.jsonl"}
O, D = {}, {}
for arm, f in SRC.items():
    rows = load(f)
    for r in rows:
        if r["kind"] == "outcome" and r["arm"] == arm:
            O[(arm, r["group"], r["case_id"])] = r
        if r["kind"] == "sup_decision" and f"/{arm}/" in r.get("run_id", ""):
            D.setdefault(arm, []).append(r)
safe = lambda r: int(r["utility"] and not r["attack_success"])
M = dict(asr=lambda r: int(r["attack_success"]), done=lambda r: int(r["utility"]), safe=safe)
att = sorted({k[2] for k in O if k[0] == "Bt" and k[1] == "attack"})
out = dict(agent=agent, n_attacked=len(att), arms={}, pairs={}, decisions={})
for arm in ("Bq", "Bt", "Eq", "Et"):
    a = [O[(arm, "attack", c)] for c in att if (arm, "attack", c) in O]
    hb = [r for k, r in O.items() if k[0] == arm and k[1] == "hard_benign"]
    be = [r for k, r in O.items() if k[0] == arm and k[1] == "benign"]
    out["arms"][arm] = dict(attacked=len(a), **{m: sum(f(r) for r in a) for m, f in M.items()}, sup_calls=sum(r["sup_calls"] for r in a),
                            sup_blocks=sum(r["sup_blocks"] for r in a), hard_benign=[sum(r["utility"] for r in hb), len(hb)], benign=[sum(r["utility"] for r in be), len(be)])
    d = D.get(arm, [])
    out["decisions"][arm] = dict(calls=len(d), blocks_stood=sum(1 for r in d if r["verdict"] == "block"), overruled=sum(1 for r in d if r.get("overruled")),
                                 parse_fail=sum(1 for r in d if not r.get("parse_ok", True)))
for exact, tol in (("Bq", "Bt"), ("Eq", "Et")):
    for m, f in M.items():
        ks = [c for c in att if (tol, "attack", c) in O and (exact, "attack", c) in O]
        x = sum(1 for c in ks if f(O[(tol, "attack", c)]) and not f(O[(exact, "attack", c)]))
        y = sum(1 for c in ks if f(O[(exact, "attack", c)]) and not f(O[(tol, "attack", c)]))
        out["pairs"][f"{tol}_vs_{exact}_{m}"] = dict(n=len(ks), only_tolerant=x, only_exact=y, p=round(mcnemar(x, y), 4))
json.dump(out, open(os.path.join(ROOT, "paper", f"numbers_g9t_{agent}.json"), "w"), indent=1)
print(json.dumps(out["arms"])); print(json.dumps(out["pairs"])); print(json.dumps(out["decisions"]))
