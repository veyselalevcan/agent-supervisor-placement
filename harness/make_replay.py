# Build the replay payload for mode g5 (post-hoc reviewer, gap G5) from stored events: python harness/make_replay.py <events.jsonl> <arms,comma> [out.b64]
# One record per run: run_id, arm, case_id, group, goal, outcome flags, steps [(step, tool, args, result, decision, reason)], final text.
# The payload is gzip+base64 so it can sit inside the one-cell Kaggle notebook (no dataset upload needed).
import ast, base64, gzip, json, sys

def build(path, arms):
    runs, order = {}, []
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        rid = e.get("run_id")
        if e["kind"] == "tool_call":
            args = e["args"]
            if isinstance(args, str):
                try:
                    args = ast.literal_eval(args)
                except Exception:
                    pass
            runs.setdefault(rid, dict(steps=[], final=""))["steps"].append(
                [e["step"], e["tool"], args, e["result"], e["decision"], e.get("reason", "")])
        elif e["kind"] == "model_call":
            try:
                p = json.loads(e["payload"].strip().splitlines()[0])
            except Exception:
                p = {}
            if "final" in p:
                runs.setdefault(rid, dict(steps=[], final=""))["final"] = str(p["final"])[:300]
        elif e["kind"] == "outcome" and e["arm"] in arms:
            r = runs.setdefault(rid, dict(steps=[], final=""))
            r.update({k: e.get(k) for k in ("run_id", "arm", "case_id", "group", "goal", "attack_success", "attacker_contact", "credential_leak", "utility")})
            order.append(rid)
    return [runs[r] for r in order]

def build_adaptive(paths):
    """B7: replay records for the stored adaptive-attacker runs (shards s6, s6s, s6p of the 4B agent). One record per try (an outcome row);
    the strategy, the try number and the strict success flag come from the t2_try event of the same run. 'arm' becomes '<strategy>:<defence>' so the
    reviewer summary separates them; attack_success is the strict adaptive predicate (the loose flag is kept as attack_loose)."""
    out = []
    for path in paths:
        recs = build(path, {"0", "Dplus", "B", "Bq", "Eq"})
        t2 = {}
        for line in open(path, encoding="utf-8"):
            e = json.loads(line)
            if e["kind"] == "t2_try":
                t2[e["run_id"]] = e
        for r in recs:
            e = t2[r["run_id"]]
            r.update(attack_loose=r["attack_success"], attack_success=int(e["success"]), strategy=e["strategy"], **{"try": int(e["try"])},
                     base_case_id=e["adaptive_id"], arm=f"{e['strategy']}:{r['arm']}", group="attack")
            out.append(r)
    return out

def encode(records):
    return base64.b64encode(gzip.compress(json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")

if __name__ == "__main__":
    if sys.argv[1] == "--adaptive":          # python harness/make_replay.py --adaptive out.b64 events1.jsonl events2.jsonl ...
        recs = build_adaptive(sys.argv[3:])
        b64 = encode(recs)
        print(len(recs), "adaptive runs,", len(b64), "base64 chars")
        open(sys.argv[2], "w").write(b64)
        sys.exit(0)
    recs = build(sys.argv[1], sys.argv[2].split(","))
    b64 = encode(recs)
    print(len(recs), "runs,", len(b64), "base64 chars")
    if len(sys.argv) > 3:
        open(sys.argv[3], "w").write(b64)
