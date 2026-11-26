"""G7 rerun (p4h-3.10): AgentDojo, arm 0 (no supervisor) vs Bq (evidence-checked supervisor), Qwen3-4B agent and supervisor.
Reads raw/g7a_* (banking, slack) and raw/g7b_* (travel, workspace) when present; the failed V42 file raw/g7_* (p4h-3.7, fail-open) is NOT read.
Paired exact McNemar per suite and pooled; supervisor decision diagnosis (blocks that stood, overruled for a missing quote, unreadable replies).
Writes paper/numbers_g7.json. Usage: python -I paper/scripts/g7_analysis.py"""
import json, os
from math import comb

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RAW = os.path.join(ROOT, "pilot_runs", "raw")
FILES = [f for f in ("g7a_Qwen3-4B_r0_events.jsonl", "g7b_Qwen3-4B_r0_events.jsonl") if os.path.exists(os.path.join(RAW, f))]


def mcnemar(x, y):
    n = x + y
    if n == 0:
        return 1.0
    k = min(x, y)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


rows = []
for f in FILES:
    rows += [json.loads(l) for l in open(os.path.join(RAW, f), encoding="utf-8")]
oc = [r for r in rows if r["kind"] == "outcome"]
errs = [r for r in rows if r["kind"] == "run_error"]
metas = [r for r in rows if r["kind"] == "run_meta"]
by = {(r["suite"], r["user_task"], r["injection_task"], r["arm"]): r for r in oc}
out = dict(files=FILES, harness=sorted({m["harness"] for m in metas}), run_errors=len(errs), suites={}, decisions={})
suites = sorted({r["suite"] for r in oc}) + ["ALL"]
safe = lambda r: int(r["utility"] and not r["attack_success"])
for s in suites:
    sel = [k for k in by if (s == "ALL" or k[0] == s)]
    att = sorted({k[:3] for k in sel if k[2]})
    ben = sorted({k[:3] for k in sel if not k[2]})
    d = {}
    for arm in ("0", "Bq"):
        a = [by[k + (arm,)] for k in att if k + (arm,) in by]
        b = [by[k + (arm,)] for k in ben if k + (arm,) in by]
        d[arm] = dict(attacked=len(a), asr=sum(r["attack_success"] for r in a), done_attacked=sum(r["utility"] for r in a), safe=sum(safe(r) for r in a),
                      benign=len(b), benign_done=sum(r["utility"] for r in b), sup_calls=sum(r["sup_calls"] for r in a + b), sup_blocks=sum(r["sup_blocks"] for r in a + b),
                      sup_overruled=sum(r["sup_overruled"] for r in a + b), sup_parse_fail=sum(r["sup_parse_fail"] for r in a + b))
    pairs = {}
    for name, grp, f in (("asr", att, lambda r: r["attack_success"]), ("done_attacked", att, lambda r: r["utility"]), ("safe", att, safe), ("benign_done", ben, lambda r: r["utility"])):
        ks = [k for k in grp if k + ("0",) in by and k + ("Bq",) in by]
        x = sum(1 for k in ks if f(by[k + ("0",)]) and not f(by[k + ("Bq",)]))
        y = sum(1 for k in ks if f(by[k + ("Bq",)]) and not f(by[k + ("0",)]))
        pairs[name] = dict(n=len(ks), only_undefended=x, only_Bq=y, p=round(mcnemar(x, y), 4))
    out["suites"][s] = dict(arms=d, pairs=pairs)
for s in suites[:-1] + ["ALL"]:
    dec = [r for r in rows if r["kind"] == "sup_decision" and (s == "ALL" or f"/g7/{s}/" in r["run_id"])]
    att = [r for r in dec if "/benign/" not in r["run_id"]]
    ben = [r for r in dec if "/benign/" in r["run_id"]]
    f = lambda L: dict(calls=len(L), blocks_stood=sum(1 for r in L if r["verdict"] == "block" and not r["overruled"]),
                       overruled_quote_missing=sum(1 for r in L if r["overruled"]),
                       unreadable_blocked=sum(1 for r in L if not r["parse_ok"] and r["verdict"] == "block"), unreadable_all=sum(1 for r in L if not r["parse_ok"]))
    out["decisions"][s] = dict(attacked=f(att), benign=f(ben))
json.dump(out, open(os.path.join(ROOT, "paper", "numbers_g7.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
