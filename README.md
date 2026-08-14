# CTP49906 Interpretability Labs

Classroom interpretability labs for Qwen2.5-Omni and Qwen3.5: **Logit Lens**, **Attention Knockout**, teacher-forced intervention scoring, and the **Jacobian Lens**. The marimo scripts pair a fixed guided run with a hypothesis-led playground rather than ending at a canned reproduction.

## Contents

- [`avllm_interpretability/`](avllm_interpretability/) — the experiments ([README](avllm_interpretability/README.md)), adapted from [ramaneswaran/avllm_interpretability](https://github.com/ramaneswaran/avllm_interpretability) ([project page](https://ramaneswaran.github.io/avllm_interpretability/))
- [`jacobian-lens/`](jacobian-lens/) — vendored [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) reference code ([paper](https://transformer-circuits.pub/2026/workspace/index.html))

## Quick start

Requires [uv](https://docs.astral.sh/uv/), **Python ≥ 3.11**, and an NVIDIA GPU with ≥16 GB VRAM. The Qwen2.5-Omni-3B weights download from Hugging Face on first run.

*Measured on an RTX 3090:* the model is 4.70 B parameters (bf16, talker freed) and occupies **8.88 GiB**; a full guided run peaks at **13.70 GiB**. It used to load a second SDPA copy for the logit lens, which peaked at 22.08 GiB and ran out of memory on a 24 GB card — so if you are working from an older checkout and see `CUDA out of memory`, that is why.

> **No GPU?** The avllm notebook has a break-glass replay mode: set `USE_PRECOMPUTED = True` near the top and every non-interactive plot renders from committed artifacts — no GPU, no 8 GB download. The two playgrounds and the teacher-forced measurement still need a GPU and say so rather than failing obscurely. This is the mitigation for the most common day-of problem, so it is worth knowing before the session, not during it.

```bash
cd avllm_interpretability
uv venv --python 3.11 --seed .venv   # 3.11+: wigglystuff 0.5.15+ requires it
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate

# Logit Lens → logit_lens_audio_token_analysis.csv
python src/logitlens_experiment.py --model_path Qwen/Qwen2.5-Omni-3B --video_path assets/02321.mp4

# Attention Knockout (rules: source,target,start_layer,end_layer)
python src/attention_knockout_experiment.py --model_path Qwen/Qwen2.5-Omni-3B --video_path assets/02321.mp4
```

Or run both from the classroom marimo notebook [`CTP49906_avllm_molab.py`](avllm_interpretability/CTP49906_avllm_molab.py) — open it in [molab](https://marimo.io/molab) with a GPU attached, or locally with `uvx marimo edit`. Its guidebook (cell tour, knockout catalog, suggested experiments) is in [`avllm_interpretability/README.md`](avllm_interpretability/README.md).

The companion [`CTP49906_jlens_molab.py`](jacobian-lens/CTP49906_jlens_molab.py) first compares the course reference Qwen3.5-4B Jacobian lens with the vanilla logit lens, then exposes prompt offset, layers, top-k, slice, filtering, and lens-estimator choices in a submit-gated research playground before an architecture-transfer synthesis. See the [`jacobian-lens` classroom guide](jacobian-lens/README.md#classroom-marimo-demo).

## Running the tests

Needs Python ≥ 3.11. Install `anywidget`, `wigglystuff`, `marimo` and
`qwen-omni-utils` for the full suite — without them the widget and
notebook-execution tests skip rather than fail, and the widget contract goes
unverified.

```bash
python -m pytest avllm_interpretability   # experiment logic, widgets, notebook replay
python -m pytest jacobian-lens/tests      # jlens library
python -m pytest tests                    # cross-notebook lint + jlens notebook smoke
```

Three layers, because each catches what the others cannot:

| Layer | What it checks | What it misses |
| --- | --- | --- |
| Unit tests | the pure functions behind every widget and metric | anything about the notebook itself |
| `marimo export script` | the dataflow graph builds: no cycles, no name defined twice | anything inside a cell body |
| Notebook execution (`app.run`) | every cell body runs, in dependency order, in the states a student reaches | GPU-only paths |

The execution tests drive the notebooks with `app.run(defs=…)`, overriding only
the cells that would clone the repo or load model weights — so they need no GPU
and no network. They exist because a cell shipped that raised `NameError` the
moment a file appeared on disk: it parsed, the graph built, and the branch was
unreachable until a student had done the thing that reached it.

```bash
# Dataflow check on a notebook — parses and orders the cells, runs none of them
uvx --with marimo==0.23.14 marimo export script \
  avllm_interpretability/CTP49906_avllm_molab.py -o /dev/null
```

## Classroom release guidance

These are teaching-only materials. Human accessibility, localization, licensing,
and research-governance review remain open.

The WP-6 bilingual instructor/student runbooks and the blinded audience-response
surface are **not in this repository yet** — they are planned, and the PRD
(`CTP49906_W7-11_PRD.md`) is the current specification for them. The student
worksheet that does exist is
[`avllm_interpretability/WORKSHEET.md`](avllm_interpretability/WORKSHEET.md); the
avllm notebook's run ledger exports rows in its format directly.
