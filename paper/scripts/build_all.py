"""Build every table and figure of the paper from the raw event logs.

Usage: python paper/scripts/build_all.py [raw_dir] [paper_dir]
Reads only `outcome` and `sup_decision` events of pilot_runs/raw/*_events.jsonl.
Writes paper/tables/*.tex, paper/figures/*.pdf and paper/numbers.json.
Short arm codes from the harness are mapped to descriptive names here (NAMES); the
paper never shows the codes.
"""
import json, math, os, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "pilot_runs", "raw")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..")
TAB, FIG = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")

NAMES = {  # harness code -> name used in the paper
    "0": "No defence", "R": "Permission rules", "C": "Injection classifier",
    "D": "Rules + classifier", "Dplus": "Rules + classifier + taint",
    "A": "Self-review", "B": "Independent supervisor", "E": "Hybrid supervisor",
    "Bf": "Independent sup. + feedback", "Ef": "Hybrid sup. + feedback",
    "Bq": "Independent sup. + evidence", "Eq": "Hybrid sup. + evidence",
}
MEM_NAMES = {"0": "No defence", "W": "Write gate", "Wf": "Write gate + feedback",
             "Wq": "Write gate + evidence", "Emem": "Write gate + signed instructions"}
GOALS = {"exfil": "Credential exfiltration", "leak": "Document leak", "fraud": "Payment fraud",
         "sabotage": "Content sabotage", "persist": "Persistence"}
SUP_Q, SUP_P = "Qwen3-4B", "Phi-4-mini"

# Which shard supplies each (agent, supervisor, arm). Anchor arms that were re-run
# (B in s3fb, Bf in s3q-a) reproduced the earlier shard in 222/222 cases; the first
# shard is used.
SOURCES = {
    "Qwen3-4B": {SUP_Q: {"0": "s2", "Dplus": "s2", "A": "s2", "B": "s2", "E": "s2",
                         "Bf": "s3fb", "Ef": "s3fb", "Bq": "s3q", "Eq": "s3q"},
                 SUP_P: {a: "s4x" for a in ("B", "E", "Bq", "Eq")}},
    "Qwen3-1.7B": {SUP_Q: {"0": "s2", "Dplus": "s2", "A": "s2", "B": "s2", "E": "s2",
                           "Bf": "s3q", "Bq": "s3q", "Eq": "s3q"},
                   SUP_P: {a: "s4x" for a in ("B", "E", "Bq", "Eq")}},
}
SHARD_FILE = {"s2": "s2_{m}_r0", "s3fb": "s3fb_{m}_r0", "s3q": "s3q_{m}_r0",
              "s4x": "s4x_{m}_r0_sup-Phi-4-mini-instruct"}

NUM = {}  # numbers quoted in the text


def load(name):
    """Outcomes per arm (dict case_id -> outcome) and overrule counts per arm."""
    out, over, over_runs = defaultdict(dict), defaultdict(int), defaultdict(set)
    with open(os.path.join(RAW, name + "_events.jsonl"), encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["kind"] == "outcome":
                assert e["case_id"] not in out[e["arm"]], e["run_id"]
                out[e["arm"]][e["case_id"]] = e
            elif e["kind"] == "sup_decision" and e.get("overruled"):
                arm = e["run_id"].split("/")[3]
                over[arm] += 1
                over_runs[arm].add(e["run_id"])
    return out, over, {a: len(r) for a, r in over_runs.items()}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(x, y, key):
    """Exact two-sided McNemar on paired cases; x, y: case_id -> outcome. Returns (b, c, p)."""
    ids = sorted(set(x) & set(y))
    b = sum(1 for i in ids if key(x[i]) and not key(y[i]))
    c = sum(1 for i in ids if key(y[i]) and not key(x[i]))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, p)


def fmt_p(p):
    if p < 1e-4:
        return r"$<10^{%d}$" % math.ceil(math.log10(p))
    return f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"


ASR = lambda r: r["attack_success"]
SAFE = lambda r: bool(r["utility"] and not r["attack_success"])
ATT = lambda d: {k: v for k, v in d.items() if v["group"] == "attack"}


def stats(rows, over=0, over_runs=0):
    att = [r for r in rows.values() if r["group"] == "attack"]
    ben = [r for r in rows.values() if r["group"] == "benign"]
    hb = [r for r in rows.values() if r["group"] == "hard_benign"]
    k = sum(r["attack_success"] for r in att)
    return dict(n_att=len(att), asr=k, ci=wilson(k, len(att)),
                ben=sum(r["utility"] for r in ben), n_ben=len(ben),
                hb=sum(r["utility"] for r in hb), n_hb=len(hb),
                util_att=sum(r["utility"] for r in att),
                safe=sum(1 for r in att if SAFE(r)),
                sup_calls=sum(r.get("sup_calls", 0) for r in rows.values()),
                sup_blocks=sum(r.get("sup_blocks", 0) for r in rows.values()),
                parse_fail=sum(r.get("sup_parse_fail", 0) for r in rows.values()),
                over=over, over_runs=over_runs,
                wall=sum(r["wall_s"] for r in rows.values()) / max(1, len(rows)),
                sup_s=sum(r.get("sup_s", 0) for r in rows.values()) / max(1, len(rows)),
                by_goal={g: (sum(r["attack_success"] for r in att if r["goal"] == g),
                             sum(1 for r in att if r["goal"] == g)) for g in GOALS})


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.5)


# ---------------------------------------------------------------- stage 1
def stage1():
    """Review B3: repeat 1 (180 attacked cases) with Wilson intervals for N = 180, the range over the three repeats, done under attack and
    safe completion; the pooled ASR (540 runs) stays as a column because the text quotes it."""
    rows, res = [], {}
    ARMS = ("0", "R", "C", "D", "Dplus")
    for m in ("Qwen3-1.7B", "Qwen3-4B"):
        per = {}
        for rep in range(3):
            out, _, _ = load(f"s1_{m}_r{rep}")
            for arm in ARMS:
                att = [r for r in out[arm].values() if r["group"] == "attack"]
                ben = [r for r in out[arm].values() if r["group"] == "benign"]
                hb = [r for r in out[arm].values() if r["group"] == "hard_benign"]
                per.setdefault(arm, []).append(dict(won=sum(r["attack_success"] for r in att), n=len(att),
                                                    done=sum(r["utility"] for r in att),
                                                    safe=sum(1 for r in att if SAFE(r)),
                                                    ben=(sum(r["utility"] for r in ben), len(ben)),
                                                    hb=(sum(r["utility"] for r in hb), len(hb))))
        res[m] = per
        for arm in ARMS:
            r1 = per[arm][0]
            won = [x["won"] for x in per[arm]]
            pk, pn = sum(won), sum(x["n"] for x in per[arm])
            lo, hi = wilson(r1["won"], r1["n"])
            rows.append(f"{m} & {NAMES[arm]} & {r1['won']}/{r1['n']} & {100*r1['won']/r1['n']:.1f} & {100*lo:.1f}--{100*hi:.1f} & {min(won)}--{max(won)} & "
                        f"{100*pk/pn:.1f} & {r1['done']} & {r1['safe']} & {r1['ben'][0]}/{r1['ben'][1]} & {r1['hb'][0]}/{r1['hb'][1]} \\\\")
        rows.append(r"\midrule")
        NUM[f"s1|{m}"] = {a: per[a] for a in per}
    write("tab_stage1.tex", r"""\begin{table*}[t]
\centering
\caption{Deterministic and classifier defences. Repeat 1 (180 attacked cases, 95\% Wilson interval for $N=180$); ``Range'' is the minimum and maximum of attacks won over the three repeats; ``Pooled'' is the attack success rate over the 540 runs of all three repeats. Done: attacked cases whose task was completed; Safe: completed and attack failed. Benign and hard-benign counts are repeat 1; this stage had 18 benign and 3 hard-benign tasks.}
\label{tab:stage1}
\footnotesize
\begin{tabular}{llrrcrrrrrr}
\toprule
Agent & Defence & Won & ASR\,\% & 95\% CI & Range & Pooled\,\% & Done & Safe & Benign & Hard-ben. \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    labels = ["No\ndefence", "Permission\nrules", "Injection\nclassifier", "Rules +\nclassifier", "Rules + clf.\n+ taint"]
    w = 0.38
    for j, (m, col) in enumerate((("Qwen3-1.7B", "#9ecae1"), ("Qwen3-4B", "#08519c"))):
        xs, ys, el, eh = [], [], [], []
        for i, a in enumerate(ARMS):
            k, n = res[m][a][0]["won"], res[m][a][0]["n"]
            lo, hi = wilson(k, n)
            xs.append(i + (j - 0.5) * w); ys.append(100 * k / n)
            el.append(100 * (k / n - lo)); eh.append(100 * (hi - k / n))
        ax.bar(xs, ys, w, color=col, edgecolor="black", linewidth=0.4, label=f"Agent {m}, repeat 1",
               yerr=[el, eh], capsize=1.5, error_kw=dict(lw=0.6))
        for rep in (1, 2):    # repeats 2 and 3 as dots
            ax.scatter(xs, [100 * res[m][a][rep]["won"] / res[m][a][rep]["n"] for a in ARMS], s=5, color="black", zorder=4,
                       label="Repeats 2 and 3" if (j == 1 and rep == 1) else None)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Attack success rate (%)", fontsize=7)
    ax.set_ylim(0, 100); ax.legend(fontsize=6, frameon=False)
    style_axes(ax)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_stage1_asr.pdf")); plt.close(fig)


# ---------------------------------------------------------------- single session, LLM defences
def single_session():
    data, cache = {}, {}
    for m, sups in SOURCES.items():
        for sup, arms in sups.items():
            for arm, shard in arms.items():
                fn = SHARD_FILE[shard].format(m=m)
                if fn not in cache:
                    cache[fn] = load(fn)
                out, over, over_runs = cache[fn]
                data[(m, sup, arm)] = (out[arm], stats(out[arm], over.get(arm, 0), over_runs.get(arm, 0)))
    order = [(SUP_Q, a) for a in ("0", "Dplus", "A", "B", "E", "Bf", "Ef", "Bq", "Eq")] + \
            [(SUP_P, a) for a in ("B", "E", "Bq", "Eq")]
    body = []
    for m in ("Qwen3-4B", "Qwen3-1.7B"):
        body.append(r"\multicolumn{10}{l}{\emph{Agent %s}}\\" % m)
        for sup, arm in order:
            if (m, sup, arm) not in data:
                continue
            s = data[(m, sup, arm)][1]
            supcol = "--" if arm in ("0", "Dplus") else ("agent itself" if arm == "A" else sup)
            over = f"{s['over']}" if arm in ("Bq", "Eq") else "--"
            calls = f"{s['sup_calls']}" if s["sup_calls"] else "--"
            body.append(f"{NAMES[arm]} & {supcol} & {s['asr']} & {s['ben']}/{s['n_ben']} & {s['hb']}/{s['n_hb']} & "
                        f"{s['util_att']} & {s['safe']} & {calls} & {over} & {s['wall']:.1f} \\\\")
            NUM[f"main|{m}|{sup}|{arm}"] = {k: v for k, v in s.items() if k != "ci"}
        body.append(r"\midrule")
    write("tab_main.tex", r"""\begin{table*}[t]
\centering
\caption{Single-session results (one repeat). Per row: 180 attacked, 18 benign and 24 hard-benign cases. All rows of one agent
share the case seeds, so rows are paired. Attacks won, done under attack and safe completion count attacked cases (of 180).
Safe completion: the user's task was done and the attack failed. Checks: LLM check calls over all 222 cases.
Overruled: blocks cancelled because the quoted evidence was not in the action. Time: mean wall-clock seconds per case on T4 GPUs.}
\label{tab:main}
\footnotesize
\begin{tabular}{llrrrrrrrr}
\toprule
Defence & Supervisor & Attacks won & Benign done & Hard-benign done & Done under attack & Safe completion & Checks & Overruled & Time (s) \\
\midrule
""" + "\n".join(body[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    tests = [
        ("No defence vs.\\ self-review", "ASR", SUP_Q, "0", SUP_Q, "A", ASR),
        ("No defence vs.\\ independent sup.", "ASR", SUP_Q, "0", SUP_Q, "B", ASR),
        ("Rules+clf.+taint vs.\\ independent sup.", "ASR", SUP_Q, "Dplus", SUP_Q, "B", ASR),
        ("Rules+clf.+taint vs.\\ independent sup.", "Safe", SUP_Q, "Dplus", SUP_Q, "B", SAFE),
        ("Independent sup.\\ vs.\\ + feedback", "Safe", SUP_Q, "B", SUP_Q, "Bf", SAFE),
        ("+ feedback vs.\\ + evidence", "Safe", SUP_Q, "Bf", SUP_Q, "Bq", SAFE),
        ("+ feedback vs.\\ + evidence", "ASR", SUP_Q, "Bf", SUP_Q, "Bq", ASR),
        ("Hybrid sup.\\ vs.\\ + evidence", "Safe", SUP_Q, "E", SUP_Q, "Eq", SAFE),
        ("Independent vs.\\ + evidence (Phi)", "Safe", SUP_P, "B", SUP_P, "Bq", SAFE),
        ("Independent vs.\\ + evidence (Phi)", "ASR", SUP_P, "B", SUP_P, "Bq", ASR),
        ("Qwen vs.\\ Phi sup., independent", "ASR", SUP_Q, "B", SUP_P, "B", ASR),
        ("Qwen vs.\\ Phi sup., + evidence", "Safe", SUP_Q, "Bq", SUP_P, "Bq", SAFE),
    ]
    body = []
    for label, metric, s1, a1, s2, a2, key in tests:
        cells = []
        for m in ("Qwen3-4B", "Qwen3-1.7B"):
            x, y = ATT(data[(m, s1, a1)][0]), ATT(data[(m, s2, a2)][0])
            b, c, p = mcnemar(x, y, key)
            kx = sum(1 for v in x.values() if key(v)); ky = sum(1 for v in y.values() if key(v))
            cells.append(f"{kx}$\\to${ky} & {b}/{c} & {fmt_p(p)}")
            NUM[f"test|{a1}-{a2}|{s1}-{s2}|{metric}|{m}"] = dict(x=kx, y=ky, b=b, c=c, p=p)
        body.append(f"{label} & {metric} & " + " & ".join(cells) + r" \\")
    write("tab_tests.tex", r"""\begin{table*}[t]
\centering
\caption{Paired exact McNemar tests on the 180 attacked cases (same seeds). ASR: attack won; Safe: safe completion.
Count: first defence $\to$ second defence; $b/c$: discordant pairs (only first / only second).}
\label{tab:tests}
\footnotesize
\begin{tabular}{llrrrrrr}
\toprule
& & \multicolumn{3}{c}{Agent Qwen3-4B} & \multicolumn{3}{c}{Agent Qwen3-1.7B} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Comparison & Metric & Count & $b/c$ & $p$ & Count & $b/c$ & $p$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    # trade-off scatter: one marker per defence (legend), filled = Qwen supervisor, open = Phi supervisor
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=True)
    style = {"0": ("o", "#000000"), "Dplus": ("s", "#777777"), "A": ("v", "#aaaaaa"),
             "B": ("o", "#d95f02"), "E": ("s", "#d95f02"), "Bf": ("^", "#b8860b"), "Ef": ("D", "#b8860b"),
             "Bq": ("o", "#1b9e77"), "Eq": ("s", "#1b9e77")}
    for ax, m in zip(axes, ("Qwen3-4B", "Qwen3-1.7B")):
        for (mm, sup, arm), (_, s) in data.items():
            if mm != m:
                continue
            mk, col = style[arm]
            ax.scatter(100 * s["asr"] / 180, 100 * s["safe"] / 180, marker=mk, s=30,
                       facecolors=col if sup == SUP_Q else "white", edgecolors=col, linewidths=1.1, zorder=3)
        ax.set_title(f"Agent {m}", fontsize=7.5)
        ax.set_xlabel("Attack success rate (%)", fontsize=7)
        ax.set_xlim(0, 95); ax.set_ylim(0, 90)
        ax.grid(alpha=0.25, lw=0.4); style_axes(ax)
    axes[0].set_ylabel("Safe completion (%)", fontsize=7)
    handles = [Line2D([], [], marker=style[a][0], ls="", mfc=style[a][1], mec=style[a][1], label=NAMES[a])
               for a in ("0", "Dplus", "A", "B", "E", "Bf", "Ef", "Bq", "Eq")]
    handles.append(Line2D([], [], marker="o", ls="", mfc="white", mec="#555", label="Phi-4-mini sup. (open)"))
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(os.path.join(FIG, "fig_tradeoff.pdf")); plt.close(fig)
    # goal heatmap
    cols = [(SUP_Q, "0"), (SUP_Q, "Dplus"), (SUP_Q, "A"), (SUP_Q, "B"), (SUP_Q, "E"), (SUP_Q, "Bq"), (SUP_Q, "Eq"),
            (SUP_P, "B"), (SUP_P, "Bq")]
    clab = ["No\ndef.", "Rules+\nclf.+taint", "Self-\nreview", "Indep.", "Hybrid", "Indep.\n+evid.", "Hybrid\n+evid.",
            "Indep.\n(Phi)", "Indep.+\nevid.(Phi)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    for ax, m in zip(axes, ("Qwen3-4B", "Qwen3-1.7B")):
        mat = [[data[(m, s, a)][1]["by_goal"][g][0] for (s, a) in cols] for g in GOALS]
        ax.imshow(mat, cmap="Greys", vmin=0, vmax=36, aspect="auto")
        for i, row in enumerate(mat):
            for j, v in enumerate(row):
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6, color="white" if v > 20 else "black")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.replace("\n", " ") for c in clab], fontsize=5.5, rotation=40, ha="right")
        ax.set_yticks(range(len(GOALS))); ax.set_yticklabels(GOALS.values(), fontsize=6)
        ax.set_title(f"Agent {m}: attacks won (of 36 per goal)", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_goals.pdf")); plt.close(fig)
    return data


# ---------------------------------------------------------------- memory poisoning
def memory():
    src = {"Qwen3-4B": {"0": "m1_Qwen3-4B_r0", "W": "m1_Qwen3-4B_r0", "Wf": "m1_Qwen3-4B_r0",
                        "Wq": "m1q_Qwen3-4B_r0", "Emem": "m1_Qwen3-4B_r0"},
           "Qwen3-1.7B": {a: "m1_Qwen3-1.7B_r0" for a in ("0", "W", "Wf", "Wq", "Emem")}}
    cache, res, body = {}, {}, []
    for m, arms in src.items():
        body.append(r"\multicolumn{11}{l}{\emph{Agent %s}}\\" % m)
        for arm, fn in arms.items():
            if fn not in cache:
                cache[fn] = load(fn)
            out, over, _ = cache[fn]
            rows = out[arm]
            att = [r for r in rows.values() if r["group"] == "attack"]
            v1 = [r for r in att if r["vector"] == "V1"]; v3 = [r for r in att if r["vector"] == "V3"]
            s = dict(asr=sum(r["attack_success"] for r in att),
                     v1=sum(r["attack_success"] for r in v1), v3=sum(r["attack_success"] for r in v3),
                     v1_task=sum(r["utility_s1"] for r in v1),
                     ben=sum(r["utility"] for r in rows.values() if r["group"] == "benign"),
                     hb=sum(r["utility"] for r in rows.values() if r["group"] == "hard_benign"),
                     safe=sum(1 for r in att if SAFE(r)), over=over.get(arm, 0),
                     safe_v1=sum(1 for r in v1 if SAFE(r)), safe_v3=sum(1 for r in v3 if SAFE(r)),     # review B5: safe completion per channel
                     done_s2=sum(r["utility_s2"] for r in att),
                     wall=sum(r["wall_s"] for r in rows.values()) / len(rows))
            res[(m, arm)] = s
            NUM[f"mem|{m}|{arm}"] = s
            body.append(f"{MEM_NAMES[arm]} & {s['asr']} & {s['v1']}/90 & {s['v3']}/90 & {s['v1_task']}/90 & "
                        f"{s['safe']} & {s['safe_v1']}/90 & {s['safe_v3']}/90 & {s['ben']}/10 & {s['hb']}/15 & {s['wall']:.1f} \\\\")
        body.append(r"\midrule")
    for m in src:
        f = lambda a: ATT(cache[src[m][a]][0][a])
        for a1, a2, key, lab in (("0", "W", SAFE, "safe"), ("Wf", "Wq", SAFE, "safe"), ("0", "Wq", SAFE, "safe"),
                                 ("0", "Wq", ASR, "asr"), ("Wq", "Emem", ASR, "asr"), ("Wf", "Wq", ASR, "asr")):
            b, c, p = mcnemar(f(a1), f(a2), key)
            NUM[f"memtest|{m}|{a1}-{a2}|{lab}"] = dict(b=b, c=c, p=p)
    write("tab_memory.tex", r"""\begin{table*}[t]
\centering
\caption{Cross-session memory poisoning. Each case has two sessions: session 1 brings a poisoned item into the note vault,
session 2 runs a clean task that reads it. Per row: 180 attacked cases (90 per poisoning channel), 10 benign, 15 hard-benign.
Session-1 task: the clipping task of session 1 was done (clipped-page channel only). Safe completion: the tasks of both sessions were done and the attack did not succeed in session 2; the last two safe columns split it by channel. The write gate is a Qwen3-4B supervisor.}
\label{tab:memory}
\footnotesize
\begin{tabular}{lrrrrrrrrrr}
\toprule
Defence & Attacks won & Clipped page & Synced note & Session-1 task & Safe completion & Safe, clipped & Safe, synced & Benign & Hard-benign & Time (s) \\
\midrule
""" + "\n".join(body[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True)
    arms = ("0", "W", "Wf", "Wq", "Emem")
    lab = ["No\ndefence", "Write\ngate", "+ feed-\nback", "+ evi-\ndence", "+ signed\ninstr."]
    series = (("v1", "Clipped page: attacks won", "#d95f02"),
              ("v1_task", "Clipped page: session-1 task done", "#1b9e77"),
              ("v3", "Synced note: attacks won", "#7570b3"))
    w = 0.26
    for ax, m in zip(axes, ("Qwen3-4B", "Qwen3-1.7B")):
        for j, (k, name, col) in enumerate(series):
            ax.bar([i + (j - 1) * w for i in range(len(arms))], [100 * res[(m, a)][k] / 90 for a in arms], w,
                   color=col, edgecolor="black", linewidth=0.3, label=name)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(lab, fontsize=6)
        ax.set_title(f"Agent {m}", fontsize=7.5); style_axes(ax)
    axes[0].set_ylabel("% of 90 cases", fontsize=7); axes[0].set_ylim(0, 105)
    axes[1].legend(fontsize=5.8, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_memory.pdf")); plt.close(fig)


def write(name, text):
    """Tables are scaled to the column or page width."""
    if name != "tab_tests.tex":   # this one fits; never scale a small table up
        text = text.replace(r"\begin{tabular}", "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}")
        text = text.replace(r"\end{tabular}", r"\end{tabular}}")
    with open(os.path.join(TAB, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
    stage1(); single_session(); memory()
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(NUM, f, indent=1, sort_keys=True, default=str)
    print("tables:", sorted(os.listdir(TAB))); print("figures:", sorted(os.listdir(FIG)))
