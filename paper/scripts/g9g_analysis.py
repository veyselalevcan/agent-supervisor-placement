"""G9: a third supervisor family (Granite-3.3-2b, shard g9g) for the evidence-checked arms Bq/Eq with the 4B agent, paired by case_id with the same arms
under the Qwen3-4B supervisor (s3q) and the Phi-4-mini supervisor (s4x). Writes paper/numbers_g9g_<agent>.json.
Usage: python -I paper/scripts/g9g_analysis.py [Qwen3-4B]"""
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


SRC = {"granite": f"g9g_{agent}_r0_sup-granite-3.3-2b-instruct_events.jsonl", "qwen": f"s3q_{agent}_r0_events.jsonl",
       "phi": f"s4x_{agent}_r0_sup-Phi-4-mini-instruct_events.jsonl"}
O, D = {}, {}
for sup, f in SRC.items():
    for r in load(f):
        if r["kind"] == "outcome" and r["arm"] in ("Bq", "Eq"):
            O[(sup, r["arm"], r["group"], r["case_id"])] = r
        if r["kind"] == "sup_decision" and r.get("run_id", "").split("/")[-3:-1] and any(f"/{a}/" in r["run_id"] for a in ("Bq", "Eq")):
            arm = "Bq" if "/Bq/" in r["run_id"] else "Eq"
            D.setdefault((sup, arm), []).append(r)
safe = lambda r: int(r["utility"] and not r["attack_success"])
M = dict(asr=lambda r: int(r["attack_success"]), done=lambda r: int(r["utility"]), safe=safe)
att = sorted({k[3] for k in O if k[0] == "granite" and k[2] == "attack"})
out = dict(agent=agent, n_attacked=len(att), arms={}, pairs={}, decisions={})
for sup in ("qwen", "phi", "granite"):
    for arm in ("Bq", "Eq"):
        a = [O[(sup, arm, "attack", c)] for c in att if (sup, arm, "attack", c) in O]
        hb = [r for k, r in O.items() if k[0] == sup and k[1] == arm and k[2] == "hard_benign"]
        be = [r for k, r in O.items() if k[0] == sup and k[1] == arm and k[2] == "benign"]
        out["arms"][f"{sup}_{arm}"] = dict(attacked=len(a), **{m: sum(f(r) for r in a) for m, f in M.items()}, sup_calls=sum(r["sup_calls"] for r in a),
                                           sup_blocks=sum(r["sup_blocks"] for r in a), hard_benign=[sum(r["utility"] for r in hb), len(hb)], benign=[sum(r["utility"] for r in be), len(be)])
        d = D.get((sup, arm), [])
        out["decisions"][f"{sup}_{arm}"] = dict(calls=len(d), blocks_stood=sum(1 for r in d if r["verdict"] == "block"), overruled=sum(1 for r in d if r.get("overruled")),
                                                parse_fail=sum(1 for r in d if not r.get("parse_ok", True)))
for arm in ("Bq", "Eq"):
    for other in ("qwen", "phi"):
        for m, f in M.items():
            ks = [c for c in att if ("granite", arm, "attack", c) in O and (other, arm, "attack", c) in O]
            x = sum(1 for c in ks if f(O[("granite", arm, "attack", c)]) and not f(O[(other, arm, "attack", c)]))
            y = sum(1 for c in ks if f(O[(other, arm, "attack", c)]) and not f(O[("granite", arm, "attack", c)]))
            out["pairs"][f"granite_vs_{other}_{arm}_{m}"] = dict(n=len(ks), only_granite=x, only_other=y, p=round(mcnemar(x, y), 4))
json.dump(out, open(os.path.join(ROOT, "paper", f"numbers_g9g_{agent}.json"), "w"), indent=1)
for k, v in out["arms"].items():
    print(k, v)
print(json.dumps(out["pairs"])); print(json.dumps(out["decisions"]))
