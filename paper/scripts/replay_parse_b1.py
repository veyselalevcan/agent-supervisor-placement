"""B1 (review 2026-11-19): replay the Qwen3-4B supervisor replies that did not parse under harness p4h-2.2 through the
repaired parser of p4h-2.3 and the evidence check, using only the stored events. Writes paper/numbers_b1.json.

Usage: python -I paper/scripts/replay_parse_b1.py   (from the project root)
What the replay can show: the decision of the repaired parser + evidence check for each stored reply.
What it cannot show: the agent's later steps. In p4h-2.2 the agent was told "the action does more than the task asked"
for an unparseable reply, while p4h-2.3 would pass the supervisor's own reason; a run with such a reply could therefore
continue differently. Those runs are listed and bounded, not re-run (no GPU)."""
import json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import p4h  # noqa: E402

RAW = os.path.join(ROOT, "pilot_runs", "raw")
OUT = os.path.join(ROOT, "paper", "numbers_b1.json")


def rows(name):
    with open(os.path.join(RAW, name + "_events.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


ev = rows("s3q_Qwen3-4B_r0")
meta = next(r for r in ev if r["kind"] == "run_meta")
assert meta["harness"] == "p4h-2.2", meta["harness"]
tool_at = {(r["run_id"], r["step"]): r for r in ev if r["kind"] == "tool_call"}
oc = {r["run_id"]: r for r in ev if r["kind"] == "outcome"}
fails = [r for r in ev if r["kind"] == "sup_decision" and r["parse_ok"] is False]

res = {"n_parse_fail_replies": len(fails), "harness_stored": meta["harness"], "harness_parser": p4h.HARNESS_VERSION}
by_run = defaultdict(list)
new_block = new_overruled = new_allow = 0
for r in fails:
    verdict, ok = p4h.parse_verdict(r["payload"])
    obj, _ = p4h.parse_sup(r["payload"])
    act = tool_at[(r["run_id"], r["step"])]
    found = p4h.evidence_found(str((obj or {}).get("evidence", "")), act["tool"], act["args"])
    assert r["verdict"] == "block" and not r["overruled"]
    if not ok:
        decision = "still unparsed (fail closed)"
    elif verdict == "block" and found:
        decision = "block"; new_block += 1
    elif verdict == "block":
        decision = "overruled to allow"; new_overruled += 1
    else:
        decision = "allow"; new_allow += 1
    by_run[r["run_id"]].append(decision)
res["replay_block"] = new_block
res["replay_overruled_to_allow"] = new_overruled
res["replay_allow"] = new_allow
res["replay_unparsed"] = len(fails) - new_block - new_overruled - new_allow
runs = []
for rid, ds in sorted(by_run.items()):
    o = oc[rid]
    runs.append({"run_id": rid, "case": o["case_id"], "arm": o["arm"], "group": o["group"], "replies": len(ds),
                 "attack_success": o["attack_success"], "utility": o["utility"]})
res["affected_runs"] = len(runs)
res["affected_by_arm"] = {a: sum(1 for x in runs if x["arm"] == a) for a in ("Bq", "Eq")}
res["affected_by_group"] = {g: sum(1 for x in runs if x["group"] == g) for g in sorted({x["group"] for x in runs})}
res["affected_cases"] = sorted({x["case"] for x in runs})
att = [x for x in runs if x["group"] == "attack"]
res["affected_attacked_runs"] = len(att)
res["affected_attacked_won"] = sum(x["attack_success"] for x in att)
res["affected_attacked_done"] = sum(x["utility"] for x in att)
res["runs"] = runs
# arm totals for context: attacked runs per arm, attacks won, safe completion (task done and no attack success)
tot = {}
for a in ("Bq", "Eq"):
    at = [o for o in oc.values() if o["arm"] == a and o["group"] == "attack"]
    tot[a] = {"n": len(at), "won": sum(o["attack_success"] for o in at),
              "safe": sum(1 for o in at if o["utility"] and not o["attack_success"])}
res["arm_totals_v22"] = tot
res["affected_runs_with_attack_or_success_possible"] = len(att)
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))
