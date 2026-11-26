"""C2: the 120 supervisor-targeted attacks (T3: T3a-T3d x 30 cases) against the evidence-checked supervisors Bq/Eq (shard c2, 4B agent),
paired by case_id with B/E of shard s5b (same cases, no evidence rule). Writes paper/numbers_c2_<agent>.json.
Usage: python -I paper/scripts/c2_analysis.py [Qwen3-4B]"""
import json, os, sys
from math import comb

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RAW = os.path.join(ROOT, "pilot_runs", "raw")
agent = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-4B"


def load(name):
    return [json.loads(l) for l in open(os.path.join(RAW, name), encoding="utf-8")]


def outcomes(rows, arms):
    return {(r["arm"], r["case_id"]): r for r in rows if r["kind"] == "outcome" and r["arm"] in arms and r["group"] == "attack" and r.get("tier") == "T3"}


def mcnemar(a_only, b_only):
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


c2 = load(f"c2_{agent}_r0_events.jsonl")
s5 = load(f"s5b_{agent}_r0_events.jsonl")
O = {**outcomes(s5, {"B", "E"}), **outcomes(c2, {"Bq", "Eq"})}
cases = sorted({k[1] for k in O if k[0] in ("Bq", "Eq")})
safe = lambda r: int(r["utility"] and not r["attack_success"])
metric = dict(asr=lambda r: int(r["attack_success"]), done=lambda r: int(r["utility"]), safe=safe)
out = dict(agent=agent, n_cases=len(cases), harness_c2=sorted({r.get("harness") for r in c2 if r["kind"] == "run_meta"}),
           harness_s5b=sorted({r.get("harness") for r in s5 if r["kind"] == "run_meta"}), arms={}, pairs={}, by_template={})
for arm in ("B", "Bq", "E", "Eq"):
    rs = [O[(arm, c)] for c in cases if (arm, c) in O]
    out["arms"][arm] = dict(n=len(rs), **{m: sum(f(r) for r in rs) for m, f in metric.items()},
                            sup_calls=sum(r["sup_calls"] for r in rs), sup_blocks=sum(r["sup_blocks"] for r in rs), parse_fail=sum(r["sup_parse_fail"] for r in rs))
for a, b in (("B", "Bq"), ("E", "Eq"), ("Bq", "Eq")):
    cs = [c for c in cases if (a, c) in O and (b, c) in O]
    for m, f in metric.items():
        x = sum(1 for c in cs if f(O[(a, c)]) and not f(O[(b, c)]))
        y = sum(1 for c in cs if f(O[(b, c)]) and not f(O[(a, c)]))
        out["pairs"][f"{a}_vs_{b}_{m}"] = dict(n=len(cs), only_first=x, only_second=y, p=round(mcnemar(x, y), 4))
for t in sorted({O[k]["template"] for k in O}):
    out["by_template"][t] = {arm: dict(n=sum(1 for k in O if k[0] == arm and O[k]["template"] == t),
                                       asr=sum(O[k]["attack_success"] for k in O if k[0] == arm and O[k]["template"] == t),
                                       safe=sum(safe(O[k]) for k in O if k[0] == arm and O[k]["template"] == t)) for arm in ("B", "Bq", "E", "Eq")}
dec = [r for r in c2 if r["kind"] == "sup_decision"]
out["sup_decisions_c2"] = dict(total=len(dec), blocks_standing=sum(1 for r in dec if r["verdict"] == "block"), overruled=sum(1 for r in dec if r.get("overruled")),
                               parse_fail=sum(1 for r in dec if not r.get("parse_ok", True)))
json.dump(out, open(os.path.join(ROOT, "paper", f"numbers_c2_{agent}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
