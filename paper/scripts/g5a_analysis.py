"""B7 (V61): post-hoc reviewer replayed over the stored adaptive runs (mode g5, stage g5a, p4h-3.11).
Usage: python -I paper/scripts/g5a_analysis.py   -> prints per-arm counts and writes paper/numbers_g5a.json"""
import collections, json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ev = [json.loads(l) for l in open(os.path.join(ROOT, "pilot_runs", "raw", "g5a_Qwen3-4B_r0_v61_events.jsonl"), encoding="utf-8")]
meta = [e for e in ev if e["kind"] == "run_meta"][0]
rv = [e for e in ev if e["kind"] == "review"]
out = {"harness": meta["harness"], "reviewer": meta.get("reviewer_model"), "n_reviews": len(rv),
       "unique_run_arm": len({(e["run_id"], e.get("arm")) for e in rv}), "parse_fail": sum(1 for e in rv if not e["parse_ok"]), "by_arm": {}}
for arm in sorted({e.get("arm") for e in rv}, key=str):
    r = [e for e in rv if e.get("arm") == arm]
    att = [e for e in r if e["attack_success"]]
    cln = [e for e in r if not e["attack_success"]]
    viol = [e for e in r if e["violations_executed"]]
    early = [e for e in viol if e["flagged"] and e["first_violation_step"] is not None and e["review_step"] is not None and e["review_step"] <= e["first_violation_step"]]
    out["by_arm"][str(arm)] = {
        "reviews": len(r), "attacks_won": len(att), "attacks_flagged": sum(1 for e in att if e["flagged"]),
        "clean_runs": len(cln), "clean_flagged": sum(1 for e in cln if e["flagged"]),
        "raw_flag_overruled": sum(1 for e in r if e["overruled"]),
        "runs_with_executed_violation": len(viol), "violation_runs_flagged": sum(1 for e in viol if e["flagged"]),
        "violation_runs_flag_at_or_before_first_violation": len(early),
    }
    print(arm, out["by_arm"][str(arm)])
json.dump(out, open(os.path.join(ROOT, "paper", "numbers_g5a.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k != "by_arm"})
