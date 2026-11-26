# Build a one-cell Kaggle notebook from p4h.py with the CONFIG for one shard.
#   python harness/make_nb.py s0 Qwen/Qwen3-4B 0 10      -> kaggle/p4h_s0_Qwen3-4B_r0.ipynb
#   python harness/make_nb.py s1 Qwen/Qwen3-1.7B 2       -> kaggle/p4h_s1_Qwen3-1.7B_r2.ipynb
#   python harness/make_nb.py s2_smoke Qwen/Qwen3-4B 0 14 -> stage 2 arms (0, Dplus, A, B, E), 14 stratified cases
#   python harness/make_nb.py m1_smoke Qwen/Qwen3-4B 0 14 -> T-mem harness p4m.py, arms (0, W, Emem)
#   python harness/make_nb.py s3fb Qwen/Qwen3-4B 0        -> feedback arms (B anchor, Bf, Ef), p4h.py
#   python harness/make_nb.py m1q Qwen/Qwen3-4B 0         -> T-mem Wq add-on (Wf anchor, Wq), p4m.py
#   python harness/make_nb.py s3q Qwen/Qwen3-4B 0         -> evidence arms (Bf anchor, Bq, Eq), p4h.py
#   python harness/make_nb.py s5 Qwen/Qwen3-4B 0 14       -> C-lite arms Cl, ClE on T1+T3 (smoke 14); s5b = B, E on T3; c4 = scripted compromise (any model arg)
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

ARMS = {"s0": ["0", "R", "C", "D", "Dplus"], "s1": ["0", "R", "C", "D", "Dplus"], "s2": ["0", "Dplus", "A", "B", "E"],
        "m1": ["0", "W", "Wf", "Wq", "Emem"],   # Wq added p4m-1.2 (2026-11-10); the 4B m1 shard ran without it
        "m1q": ["Wf", "Wq"],                    # 4B add-on for Wq; Wf reruns as the anchor for m1-a (L45)
        "s3fb": ["B", "Bf", "Ef"],   # stage 3 block-with-feedback; B reruns as the reproducibility anchor for stage 2
        "s3q": ["Bf", "Bq", "Eq"],
        "s4x": ["B", "Bq", "E", "Eq"],
        "s5": ["Cl", "ClE"],      # p4h-3.0 C-lite (separate supervisor process), T1 + T3 cases (G1, G3); paired with s2 B/E on T1
        "s5b": ["B", "E"],        # B/E on the T3 cases only (T1 for B/E already exists in s2); separate shard: one supervisor model copy per shard
        "c4": [],
        "g9g": ["Bq", "Eq"],      # p4h-3.7 G9: Granite-3.3-2b as the supervisor (pass sup=ibm-granite/granite-3.3-2b-instruct), paired with Qwen Bq/Eq (s3q) and Phi (s4x)
        "g9t": ["Bt", "Et"],      # p4h-3.7 G9: tolerant evidence match with the Qwen supervisor, paired with Bq/Eq (s3q, repeat 0)
        "g7smoke": ["0", "Bq"],   # G7 smoke: banking + slack, 2 user tasks, 1 injection task per suite (16 runs)
        "g7a": ["0", "Bq"],       # p4h-3.10 G7 rerun, notebook 1: banking + slack (fail-closed adapter, prompt cap)
        "g7b": ["0", "Bq"],       # p4h-3.10 G7 rerun, notebook 2: travel + workspace
        "g7": ["0", "Bq"],        # p4h-3.6 AgentDojo adapter (gap G7): no supervisor vs evidence-checked supervisor
        "g5a": ["0", "Dplus", "B", "Bq", "Eq"],   # B7 (p4h-3.9): the same reviewer on the stored adaptive-attacker runs; payload arms are strategy:defence
        "g5": ["0", "R"],         # p4h-3.5 post-hoc reviewer on stored stage-1 transcripts (arms 0, R; 4B agent), gap G5
        "s6": ["0", "Dplus", "B", "Bq", "Eq"],   # p4h-3.1 adaptive attacker, free strategy (G2)
        "s6s": ["Bq", "Eq"],                     # strategy source: push the supervisor to quote the document (L52)
        "s6p": ["Bq", "Eq"],                     # p4h-3.2 strategy split: break up / re-encode the critical value so the quote is not found
        "s6e": ["Bq", "Eq"],
        # review round 2026-11-23 (p4h-3.8 / p4m-1.4)
        "c1": ["Ef"],             # C1: hybrid supervisor + feedback with the 1.7B agent (pairs with Eq of s3q and E of s2)
        "c2": ["Bq", "Eq"],       # C2: the 120 supervisor-targeted attacks (T3) against the evidence-checked supervisors, 4B agent (pairs with s5b B/E)
        "c3": ["Bn", "Bnq"],      # C3: data-blind independent supervisor, plain and + evidence (pairs with B of s2 and Bq of s3q)
        "c4p": [],                # C4: scripted memory scan with ptrace_scope 0 if the container lets us set it (CPU only)
        "m1s": ["Wqs"],           # C5: write gate + evidence + signed instructions (pairs with Wq of m1q and Emem of m1), p4m
        "d3": ["Bq", "Eq"]}       # D3: a larger supervisor (Qwen3-8B, 4-bit) on the second T4, 4B agent (pairs with s3q Bq/Eq)                     # strategy echo: force valid quotes, deny the task                 # scripted compromise of the supervisor (G4), no agent model, runs on CPU   # cross-family supervisor (DECISIONS 2026-11-10); paired with Qwen-supervisor s2 (B, E) and s3q (Bq, Eq)
#   python harness/make_nb.py s4x Qwen/Qwen3-4B 0 - <supervisor repo>   -> kaggle/p4h_s4x_Qwen3-4B_r0_sup-<name>.ipynb   # stage 3 evidence-checked blocks; Bf reruns as the anchor for s3fb (L45)

EXTRA = {"g5a": 'mode="g5"', "s5": 'cases="t1+t3"', "s5b": 'cases="t3"', "c4": 'mode="g4"', "g5": 'mode="g5"', "g7": 'mode="g7", g7_users=12, g7_injs=3',
         "g7a": 'mode="g7", g7_users=12, g7_injs=3, g7_suites=["banking", "slack"], g7_max_input_tokens=6000',
         "g7b": 'mode="g7", g7_users=12, g7_injs=3, g7_suites=["travel", "workspace"], g7_max_input_tokens=6000',
         "g7smoke": 'mode="g7", g7_users=2, g7_injs=1, g7_suites=["banking", "slack"]',
         "s6": 'mode="t2", t2_strategy="free"', "s6s": 'mode="t2", t2_strategy="source"', "s6e": 'mode="t2", t2_strategy="echo"',
         "s6p": 'mode="t2", t2_strategy="split"',
         "c2": 'cases="t3"', "d3": 'sup_quant="4bit"',
         "c4p": 'mode="g4", g4_deployments=["proc_sameuid"], g4_variants=["read_memory/scan_memory"], g4_try_ptrace0=True, g4_repeats=5'}   # extra CONFIG lines per stage (p4h-3.0)

TEST_OUT = os.path.join(__import__("tempfile").gettempdir(), "p4_nb_tests")   # tests build here, never into kaggle/ (they delete what they build)

def build(stage, model, repeat, limit=None, sup=None, out_dir=None):
    base = "p4m" if stage.startswith("m") else "p4h"
    src = open(os.path.join(HERE, f"{base}.py"), encoding="utf-8").read()
    arms = ARMS[stage.split("_")[0]]          # e.g. "s2" or "s2_smoke"
    sup_line = f'    sup_model="{sup}",\n' if sup else ""
    extra = EXTRA.get(stage.split("_")[0])
    extra_line = f"    {extra},\n" if extra else ""
    config = (f'CONFIG = dict(\n    stage="{stage}",\n    agent_model="{model}",\n    repeat={repeat},\n'
              f'    arms={json.dumps(arms)},\n    limit_cases={limit},\n{sup_line}{extra_line})\n')
    src, n = re.subn(r"CONFIG = dict\(\n.*?\n\)\n", lambda m: config, src, count=1, flags=re.S)
    assert n == 1, "CONFIG block not found"
    if stage.split("_")[0] in ("g5", "g5a"):   # embed the stored transcripts (make_replay.py payload) in the one-cell notebook
        payload = open(os.path.join(HERE, "replay_adaptive_4B_r0.b64" if stage.startswith("g5a") else "replay_s1_4B_r0.b64")).read().strip()
        assert src.count('REPLAY_B64 = ""') == 1
        src = src.replace('REPLAY_B64 = ""', f'REPLAY_B64 = "{payload}"')
    nb = dict(cells=[dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                          source=src.splitlines(keepends=True))],
              metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
              nbformat=4, nbformat_minor=4)
    tag = f"_sup-{sup.split('/')[-1]}" if sup else ""
    out_dir = out_dir or os.path.join(HERE, "..", "kaggle")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{base}_{stage}_{model.split('/')[-1]}_r{repeat}{tag}.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return os.path.normpath(path)

if __name__ == "__main__":
    a = sys.argv[1:]
    limit = int(a[3]) if len(a) > 3 and a[3] != "-" else None
    print(build(a[0], a[1], int(a[2]), limit, a[4] if len(a) > 4 else None))
