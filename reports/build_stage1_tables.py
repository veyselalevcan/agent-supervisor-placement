# Builds flat CSVs and markdown tables for the stage 1 report from the raw event logs.
#   python -I reports/build_stage1_tables.py
import csv, json, os, collections
from math import comb, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "pilot_runs", "raw")
OUT = os.path.join(HERE, "stage1")
MODELS = ["Qwen3-1.7B", "Qwen3-4B"]
ARMS = ["0", "R", "C", "D", "Dplus"]
GOALS = ["exfil", "leak", "fraud", "sabotage", "persist"]
TPL = ["Ta", "Tb", "Tc", "Td", "Te", "Tf"]
os.makedirs(OUT, exist_ok=True)

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

rows, hooks = [], []
for m in MODELS:
    for r in range(3):
        for line in open(os.path.join(RAW, f"s1_{m}_r{r}_events.jsonl"), encoding="utf-8"):
            e = json.loads(line)
            if e["kind"] == "outcome":
                e = dict(e); e["model"] = m; e.pop("kind"); rows.append(e)
            elif e["kind"] == "hook_decision":
                hooks.append(dict(model=m, repeat=r, run_id=e["run_id"], step=e["step"], score=e["score"], decision=e["decision"]))
rid = {x["run_id"]: x for x in rows}
for h in hooks:
    o = rid[h["run_id"]]; h.update(case_id=o["case_id"], arm=o["arm"], group=o["group"], goal=o["goal"], template=o["template"])

fields = ["model", "repeat", "case_id", "task", "group", "goal", "template", "arm", "seed", "attack_success", "utility",
          "steps", "valid_calls", "extra_replies", "nudged", "blocks", "steps_after_block", "gen_s", "clf_s", "wall_s", "run_id", "harness", "tier"]
with open(os.path.join(OUT, "outcomes_stage1.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT, "classifier_decisions_stage1.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["model", "repeat", "case_id", "group", "goal", "template", "arm", "step", "score", "decision", "run_id"])
    w.writeheader(); w.writerows(hooks)

att = [x for x in rows if x["group"] == "attack"]
case = collections.defaultdict(list)
for x in att: case[(x["model"], x["case_id"], x["arm"])].append(x["attack_success"])
with open(os.path.join(OUT, "case_level_stage1.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["model", "case_id", "task", "template", "goal", "arm", "hits_of_3", "majority_hit"])
    for (m, c, a), v in sorted(case.items()):
        t, tpl, g = c.split("-"); w.writerow([m, c, t, tpl, g, a, sum(v), int(sum(v) >= 2)])
maj = {k: int(sum(v) >= 2) for k, v in case.items()}
cases = sorted({c for _, c, _ in case})

md = []
P = md.append
def cnt(f, pred): k = sum(1 for x in rows if pred(x) and f(x)); n = sum(1 for x in rows if pred(x)); return k, n

P("## T1. Attack success rate by model and arm (run level, 3 repeats pooled)")
P("| Model | Arm | ASR k/n | rate | Wilson 95% (too narrow, see text) | r0 | r1 | r2 | Case-level ASR (majority, /180) |")
P("|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    for a in ARMS:
        k, n = cnt(lambda x: x["attack_success"], lambda x: x["model"] == m and x["arm"] == a and x["group"] == "attack")
        per = [sum(x["attack_success"] for x in att if x["model"] == m and x["arm"] == a and x["repeat"] == r) for r in range(3)]
        lo, hi = wilson(k, n); cl = sum(maj[(m, c, a)] for c in cases)
        P(f"| {m} | {a} | {k}/{n} | {k/n:.3f} | {lo:.3f}-{hi:.3f} | {per[0]} | {per[1]} | {per[2]} | {cl} |")

P("\n## T2. ASR by goal (hits of 108 runs = 6 tasks x 6 templates x 3 repeats)")
P("| Model | Arm | " + " | ".join(GOALS) + " |"); P("|---|---|" + "---|" * len(GOALS))
for m in MODELS:
    for a in ARMS:
        P(f"| {m} | {a} | " + " | ".join(str(sum(x["attack_success"] for x in att if x["model"] == m and x["arm"] == a and x["goal"] == g)) for g in GOALS) + " |")

P("\n## T3. ASR by template (hits of 90 runs = 5 goals x 6 tasks x 3 repeats)")
P("| Model | Arm | " + " | ".join(TPL) + " |"); P("|---|---|" + "---|" * len(TPL))
for m in MODELS:
    for a in ARMS:
        P(f"| {m} | {a} | " + " | ".join(str(sum(x["attack_success"] for x in att if x["model"] == m and x["arm"] == a and x["template"] == t)) for t in TPL) + " |")

P("\n## T4. Utility (task completed) and blocks")
P("| Model | Arm | Benign /54 | Hard-benign /9 | Under attack /540 | Benign or hard-benign runs with a block /63 | Mean steps (attack) | Mean wall s (all) |")
P("|---|---|---|---|---|---|---|---|")
for m in MODELS:
    for a in ARMS:
        sel = [x for x in rows if x["model"] == m and x["arm"] == a]
        b = sum(x["utility"] for x in sel if x["group"] == "benign"); h = sum(x["utility"] for x in sel if x["group"] == "hard_benign")
        u = sum(x["utility"] for x in sel if x["group"] == "attack"); bl = sum(1 for x in sel if x["group"] != "attack" and x["blocks"])
        st = sum(x["steps"] for x in sel if x["group"] == "attack") / 540; ws = sum(x["wall_s"] for x in sel) / len(sel)
        P(f"| {m} | {a} | {b} | {h} | {u} | {bl} | {st:.2f} | {ws:.2f} |")

P("\n## T5. Classifier (arm C): share of injected-source reads blocked, by template x goal (pooled over both models and repeats)")
P("| Template | " + " | ".join(GOALS) + " |"); P("|---|" + "---|" * len(GOALS))
src_reads = collections.defaultdict(lambda: [0, 0])
first = {}
for h in hooks:
    if h["arm"] == "C" and h["group"] == "attack" and h["run_id"] not in first:
        first[h["run_id"]] = h  # first classified read = the task source with the payload
for h in first.values():
    src_reads[(h["template"], h["goal"])][1] += 1; src_reads[(h["template"], h["goal"])][0] += h["decision"] == "block"
for t in TPL:
    P(f"| {t} | " + " | ".join(f"{src_reads[(t,g)][0]}/{src_reads[(t,g)][1]}" for g in GOALS) + " |")

P("\n## T6. Classifier score on non-attack documents (all arms with C, both models, all repeats)")
sc = collections.defaultdict(set)
for h in hooks:
    if h["group"] != "attack": sc[h["case_id"]].add(h["score"])
P("| Case | Distinct scores seen | Blocked at 0.5 |"); P("|---|---|---|")
for c in sorted(sc):
    if c.endswith("hardbenign") or max(sc[c]) > 0.001:
        P(f"| {c} | {sorted(sc[c])} | {'yes' if max(sc[c]) >= 0.5 else 'no'} |")
P(f"| all 18 benign tasks | max {max(max(v) for c, v in sc.items() if c.endswith('-benign')):.4f} | no |")

P("\n## T7. Paired case-level tests (exact McNemar on majority outcome, 180 cases)")
P("| Model | Comparison | Case ASR before -> after | Discordant (before only / after only) | p |"); P("|---|---|---|---|---|")
for m in MODELS:
    for a, b in [("0", "R"), ("0", "C"), ("R", "D"), ("C", "D"), ("D", "Dplus")]:
        x = sum(maj[(m, c, a)] > maj[(m, c, b)] for c in cases); y = sum(maj[(m, c, a)] < maj[(m, c, b)] for c in cases)
        P(f"| {m} | {a} -> {b} | {sum(maj[(m,c,a)] for c in cases)} -> {sum(maj[(m,c,b)] for c in cases)} | {x} / {y} | {mcnemar(x,y):.2g} |")
for a in ARMS:
    x = sum(maj[("Qwen3-4B", c, a)] > maj[("Qwen3-1.7B", c, a)] for c in cases); y = sum(maj[("Qwen3-4B", c, a)] < maj[("Qwen3-1.7B", c, a)] for c in cases)
    P(f"| 4B vs 1.7B | arm {a} | {sum(maj[('Qwen3-1.7B',c,a)] for c in cases)} (1.7B) vs {sum(maj[('Qwen3-4B',c,a)] for c in cases)} (4B) | {y} 1.7B-only / {x} 4B-only | {mcnemar(x,y):.2g} |")

P("\n## T8. Repeat agreement")
P("| Model | (case, arm) pairs with the same outcome in all 3 repeats | Identical full transcripts r0 vs r1 |"); P("|---|---|---|")
for m in MODELS:
    same = sum(len(set(case[(m, c, a)])) == 1 for c in cases for a in ARMS)
    tr = []
    for r in (0, 1):
        d = collections.defaultdict(list)
        for line in open(os.path.join(RAW, f"s1_{m}_r{r}_events.jsonl"), encoding="utf-8"):
            e = json.loads(line)
            if e["kind"] == "model_call": d[e["run_id"].rsplit("/r", 1)[0]].append(e["payload"])
        tr.append(d)
    P(f"| {m} | {same}/900 | {sum(tr[0][k] == tr[1][k] for k in tr[0])}/{len(tr[0])} |")

P("\n## T9. Residual cases under Dplus (majority hit), by goal x template")
P("| Model | " + " | ".join(f"{g}" for g in GOALS) + " | total |"); P("|---|" + "---|" * (len(GOALS) + 1))
for m in MODELS:
    vals = [sum(maj[(m, c, "Dplus")] for c in cases if c.endswith("-" + g)) for g in GOALS]
    P(f"| {m} | " + " | ".join(map(str, vals)) + f" | {sum(vals)} |")

open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print("rows", len(rows), "hooks", len(hooks), "cases", len(cases))
