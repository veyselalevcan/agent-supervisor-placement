"""Builds the tables of the added studies (C2, C3, G7, supervisor models G9g/D3 and tolerant match G9t, C5) from paper/numbers_*.json only.
Usage: python -I paper/scripts/build_results_tables.py   -> paper/tables/tab_c2.tex tab_c3.tex tab_g7.tex tab_sup.tex tab_c5.tex"""
import json, os

PAPER = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
J = lambda n: json.load(open(os.path.join(PAPER, n), encoding="utf-8"))
W = lambda name, s: open(os.path.join(PAPER, "tables", name), "w", encoding="utf-8", newline="\n").write(s)

HEAD = "\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{%s}\n\\footnotesize\n\\resizebox{\\linewidth}{!}{%%\n"
TAIL = "}\n\\end{table}\n"

# ---- C2: attacks aimed at the supervisor against evidence-checked supervisors
c2 = J("numbers_c2_Qwen3-4B.json")
names = {"B": "Independent supervisor", "Bq": "Independent sup. + evidence", "E": "Hybrid supervisor", "Eq": "Hybrid sup. + evidence"}
s = HEAD % ("Attacks aimed at the supervisor (120 cases, Qwen3-4B agent and supervisor). The four last columns give attacks won of 30 per kind of note.", "tab:c2")
s += "\\begin{tabular}{lrrrrrrr}\n\\toprule\nDefence & Attacks won & Done & Safe & Approval claim & Reply ``allow'' & Fake policy & Hidden comment \\\\\n\\midrule\n"
for a in ("B", "Bq", "E", "Eq"):
    r, t = c2["arms"][a], c2["by_template"]
    s += f"{names[a]} & {r['asr']} & {r['done']} & {r['safe']} & " + " & ".join(str(t[k][a]["asr"]) for k in ("T3a", "T3b", "T3c", "T3d")) + " \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n" + TAIL
W("tab_c2.tex", s)

# ---- C3: data-blind supervisor
rows = []
s = HEAD % ("A supervisor that does not see the data the agent read (blind), against the usual one (sighted); 180 attacked cases per row. Clean blocks: attacked runs in which a block that stood hit a clean e-mail to a recipient the task names. Hard-benign: tasks done of 24.", "tab:c3")
s += "\\begin{tabular}{llrrrrrr}\n\\toprule\nAgent & Supervisor & Attacks won & Done & Safe & Clean blocks & Hard-benign \\\\\n\\midrule\n"
for ag, f in (("Qwen3-4B", "numbers_c3_Qwen3-4B.json"), ("Qwen3-1.7B", "numbers_c3_Qwen3-1.7B.json")):
    d = J(f)
    for key, lab_s, lab_b in (("Bn vs B", "Independent, sighted", "Independent, blind"), ("Bnq vs Bq", "Independent + evidence, sighted", "Independent + evidence, blind")):
        x = d[key]
        s += f"{ag} & {lab_s} & {x['won_sighted']} & {x['done_sighted']} & {x['safe_sighted']} & {x['clean_block_runs_sighted']} & {x['hard_sighted'][0]} \\\\\n"
        s += f"{ag} & {lab_b} & {x['won_blind']} & {x['done_blind']} & {x['safe_blind']} & {x['clean_block_runs_blind']} & {x['hard_blind'][0]} \\\\\n"
    s += "\\midrule\n" if ag == "Qwen3-4B" else ""
s += "\\bottomrule\n\\end{tabular}\n" + TAIL
W("tab_c3.tex", s)

# ---- G7: AgentDojo
g7 = J("numbers_g7.json")
s = HEAD % ("AgentDojo (version 1.2.2, 12 user tasks per suite, 3 injection tasks; Qwen3-4B agent and supervisor): no defence (0) against the independent supervisor + evidence (Bq). Attacks, done and safe count the 36 attacked runs per suite and arm, benign the 12 runs without attack. Blocks: stood (of which blocked because the reply was unreadable) / cancelled for a missing quote.", "tab:g7")
s += "\\begin{tabular}{llrrrrr}\n\\toprule\nSuite & Arm & Attacks won & Done & Safe & Benign done & Blocks (stood (unreadable) / cancelled) \\\\\n\\midrule\n"
for su in ("banking", "slack", "travel", "workspace", "ALL"):
    for arm in ("0", "Bq"):
        r = g7["suites"][su]["arms"][arm]
        dec = g7["decisions"][su]["attacked"] if arm == "Bq" else None
        blk = f"{dec['blocks_stood']} ({dec['unreadable_blocked']}) / {dec['overruled_quote_missing']}" if dec else "--"
        label = {"ALL": "All four"}.get(su, su.capitalize())
        s += f"{label} & {'none' if arm == '0' else 'Bq'} & {r['asr']}/{r['attacked']} & {r['done_attacked']} & {r['safe']} & {r['benign_done']}/{r['benign']} & {blk} \\\\\n"
    s += "\\midrule\n" if su != "ALL" else ""
s += "\\bottomrule\n\\end{tabular}\n" + TAIL
W("tab_g7.tex", s)

# ---- supervisor models (G9g Granite, D3 Qwen3-8B 4-bit) and tolerant match (G9t)
g9g, g9t, d3 = J("numbers_g9g_Qwen3-4B.json"), J("numbers_g9t_Qwen3-4B.json"), J("numbers_d3.json")
s = HEAD % ("Other supervisors for the Qwen3-4B agent (180 attacked, 18 benign, 24 hard-benign cases; one repeat). Unreadable: replies the harness could not parse. Phi-4-mini rows repeat Table~\\ref{tab:main}. The 4-bit row loads Qwen3-8B in 4-bit precision.", "tab:sup")
s += "\\begin{tabular}{llrrrrrr}\n\\toprule\nSupervisor & Rule & Attacks won & Done & Safe & Benign done & Hard-benign done & Unreadable \\\\\n\\midrule\n"
fam = (("Qwen3-4B", "qwen", None), ("Phi-4-mini", "phi", None), ("Granite-3.3-2B", "granite", None))
for lab, key, _ in fam:
    for arm, rule in (("Bq", "Independent + evidence"), ("Eq", "Hybrid + evidence")):
        r = g9g["arms"][f"{key}_{arm}"]
        d = g9g["decisions"][f"{key}_{arm}"]
        s += f"{lab} & {rule} & {r['asr']} & {r['done']} & {r['safe']} & {r['benign'][0]}/{r['benign'][1]} & {r['hard_benign'][0]}/{r['hard_benign'][1]} & {d['parse_fail']} \\\\\n"
    if key == "qwen":
        for arm, rule in (("Bt", "Independent + tolerant match"), ("Et", "Hybrid + tolerant match")):
            r = g9t["arms"][arm]
            d = g9t["decisions"][arm]
            s += f"Qwen3-4B & {rule} & {r['asr']} & {r['done']} & {r['safe']} & {r['benign'][0]}/{r['benign'][1]} & {r['hard_benign'][0]}/{r['hard_benign'][1]} & {d['parse_fail']} \\\\\n"
s += "\\midrule\n"
for arm, rule in (("Bq", "Independent + evidence"), ("Eq", "Hybrid + evidence")):
    a, b, h = (d3["arms"][f"sup8b:{arm}:{g}"] for g in ("attack", "benign", "hard_benign"))
    s += f"Qwen3-8B, 4-bit & {rule} & {a['asr']} & {a['done']} & {a['safe']} & {b['done']}/{b['n']} & {h['done']}/{h['n']} & {a['parse_fail'] + b['parse_fail'] + h['parse_fail']} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n" + TAIL
W("tab_sup.tex", s)

# ---- C5: memory write gate with evidence and signed instructions (agent Qwen3-4B)
c5 = J("numbers_c5_Qwen3-4B.json")
s = HEAD % ("Memory write gate with evidence and signed instructions combined (agent Qwen3-4B; 180 attacked, 10 benign, 15 hard-benign cases).", "tab:c5")
s += "\\begin{tabular}{lrrrrrr}\n\\toprule\nDefence & Attacks won & Done & Safe & Benign & Hard-benign & Blocks cancelled \\\\\n\\midrule\n"
for arm, lab in (("W", "Write gate"), ("Wq", "Write gate + evidence"), ("Emem", "Write gate + signed instructions"), ("Wqs", "Write gate + evidence + signed instructions")):
    r = c5["arms"][arm]
    s += f"{lab} & {r['asr']} & {r['done']} & {r['safe']} & {r['benign'][0]}/{r['benign'][1]} & {r['hard_benign'][0]}/{r['hard_benign'][1]} & {r['sup_overruled']} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n" + TAIL
W("tab_c5.tex", s)
print("tables written")
