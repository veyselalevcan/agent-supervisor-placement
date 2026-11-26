# Where Should the Supervisor of an LLM Agent Sit?

Code, unit tests, experiment notebooks, and table-generation scripts for the paper **"Where Should the Supervisor of an LLM Agent Sit?"** by Veysel Alevcan, Paolo Calciati, Nuno Mateus-Coelho, and Luís Miguel Campos.

- **Raw Event Logs Dataset (22,716 runs, 242,950 events):** Archived on Zenodo ([doi:10.5281/zenodo.24557265](https://doi.org/10.5281/zenodo.24557265)).

---

## Study Overview & Key Findings

Tool-using LLM agents that read untrusted documents, web pages, or notes are exposed to indirect prompt injection. A common defence is a runtime supervisor that inspects actions before execution, yet **where the supervisor sits** and **what its blocks cost in task completion** are rarely measured on a single controlled harness.

Using shared random seeds, paired exact McNemar tests, and open-weight local models across **222 cases per single-session configuration** (18 benign, 24 hard-benign, 180 attacked) plus cross-session memory, adaptive attacker, process-isolation, and AgentDojo evaluations, our main findings are:

1. **Self-review fails:** A second pass by the agent's own model inside the same conversation approves most injected actions (149 vs. 155/180 attacks won for `Qwen3-4B`; `Qwen3-1.7B` approved all 276 checked actions).
2. **Independent supervisors suffer from context contamination:** Moving the supervisor to its own clean context cuts attack success from **86% (155/180)** to **22% (39/180)** for `Qwen3-4B`, but **safe completion drops to 8% (15/180)** because the supervisor reads the injected document and keeps blocking clean retries. Returning the block reason to the agent does not fix this (16/180).
3. **Evidence-checked blocks restore task completion:** Requiring the supervisor to quote the offending text from the proposed action—and cancelling blocks in code when the quote is absent—raises safe completion from **8% to 77% (139/180)** with `Qwen3-4B` and from **4% to 84% (151/180)** with `Phi-4-mini`, without a significant rise in attacks won. The rule also repairs a memory write gate in a two-session note-vault study (safe completion **57 → 134/180**).
4. **Hybrid placement closes fail-open paths:** Placing deterministic permission rules and taint checks before an evidence-checked supervisor stops credential exfiltration and leaks that a standalone LLM supervisor can miss, including under an adaptive attacker that tries to make the supervisor quote the injected document.
5. **Process and OS-user isolation matter under compromise:** Against 195 scripted compromise trials with agent privileges, a supervisor in the agent's process fell in **25/65** trials, in its own process under the same OS user in **15/65** (policy-file edits), and under a separate OS user in **0/65**.
6. **Residual risks & external validity:** Content sabotage inside the user's permissions (e.g., an altered deadline in an allowed e-mail) remains the largest residual risk (**58–78%** success). A post-hoc reviewer detects most attacks (**152/155**) but often after the first violating action has already executed. On **AgentDojo**, the evidence-checked supervisor lowers attacks won (**42 → 32/144**) but does not raise safe completion due to weak base task capability and fail-closed parse blocks.

---

## Repository Structure

| Path | Contents |
|---|---|
| `harness/p4h.py` | Single-session harness: tasks, attacks, defence arms (`0`, `R`, `C`, `D`, `Dplus`, `A`, `B`, `Bf`, `Bq`, `E`, `Ef`, `Eq`, `Bn`, `Bnq`, `En`, `Enq`, `Cl`, `ClE`), adaptive mode, process-isolation & compromise trials, logging |
| `harness/p4m.py` | Two-session cross-session memory-poisoning harness (40-note synthetic vault + instruction file; arms `0`, `W`, `Wf`, `Wq`, `Emem`, `Wqs`) |
| `harness/g5_block.py` | Post-hoc reviewer replay module |
| `harness/g7_block.py` | AgentDojo adapter (`0.1.35`, benchmark `1.2.2`) |
| `harness/make_nb.py` | Generates self-contained Kaggle notebooks for any stage, model, and repeat |
| `harness/make_replay.py` | Builds base64 replay payloads from stored event logs |
| `harness/tests/` | Offline unit tests using scripted models (no GPU or network required) |
| `kaggle/` | Exact Kaggle notebooks executed for the study |
| `paper/scripts/` | Deterministic analysis and table/figure generation scripts (`build_all.py`, `build_new_tables.py`, `build_results_tables.py`, `*_analysis.py`) |
| `reports/` | Stage 1 & Stage 2 table-generation scripts |

---

## Reproducing Results

### 1. Run Offline Unit Tests (CPU only)
The test suite uses scripted models and requires neither a GPU nor API keys:
```bash
python -m unittest discover -s harness/tests
```
*(AgentDojo adapter tests are skipped automatically if `agentdojo` is not installed; `proc_isolated` tests require `setuid` permissions.)*

### 2. Rebuild Paper Tables & Numbers from Raw Logs
1. Download `llm_agent_supervisor_placement_raw_logs.zip` from Zenodo ([doi:10.5281/zenodo.24557265](https://doi.org/10.5281/zenodo.24557265)).
2. Verify checksums:
   ```bash
   sha256sum -c SHA256SUMS.txt
   ```
3. Place the extracted `raw/` directory at `pilot_runs/raw/` in the repository root.
4. Run the analysis and table scripts in `paper/scripts/`:
   ```bash
   python paper/scripts/build_all.py
   python paper/scripts/build_new_tables.py
   python paper/scripts/build_results_tables.py
   ```

### 3. Re-run Experiments on GPU (Kaggle T4 x2)
Generate a notebook for any stage, agent, and repeat:
```bash
python harness/make_nb.py s3q Qwen/Qwen3-4B 0
```
Upload and run the generated `.ipynb` from `kaggle/` on a T4 x2 GPU environment (`Save and Run All`). Each run logs `run_meta` (harness SHA-256, Hugging Face model commit hashes, library versions, GPU) as the first line of `<shard>_events.jsonl`.

---

## Models Used

- **Agents:** `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` (non-thinking mode)
- **Supervisors:** `Qwen/Qwen3-4B`, `microsoft/Phi-4-mini-instruct`, `ibm-granite/granite-3.3-2b-instruct`, `Qwen/Qwen3-8B` (4-bit)
- **Injection Classifier:** `protectai/deberta-v3-base-prompt-injection-v2`

All models are open-weight and run locally; no commercial API was used.
