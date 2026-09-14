# CTP49906 Interpretability Labs

Classroom interpretability labs for Qwen2.5-Omni and Qwen3.5: **Logit Lens**, **Attention Knockout**, teacher-forced intervention scoring, and the **Jacobian Lens**. The marimo scripts pair a fixed guided run with a hypothesis-led playground rather than ending at a canned reproduction.

## Contents

- [`avllm_interpretability/`](avllm_interpretability/) — the experiments ([README](avllm_interpretability/README.md)), adapted from [ramaneswaran/avllm_interpretability](https://github.com/ramaneswaran/avllm_interpretability) ([project page](https://ramaneswaran.github.io/avllm_interpretability/))
- [`jacobian-lens/`](jacobian-lens/) — vendored [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) reference code ([paper](https://transformer-circuits.pub/2026/workspace/index.html))
- [`feedback-atlas/`](feedback-atlas/) — the real-time classroom feedback map ([README](feedback-atlas/README.md) · [한국어](feedback-atlas/README_kr.md)), a small web app rather than a notebook: attendees write during a presentation, opinions are embedded locally and projected to a shared 2D map

## Quick start

For **Creative AI: Creation & Practice**, start with the updated [Korean Molab notebook](avllm_interpretability/CTP49906_avllm_molab_kr.py), [instructor guide](avllm_interpretability/CLASSROOM_GUIDE_kr.md), and [student worksheet](avllm_interpretability/WORKSHEET_kr.md). In [Molab notebooks](https://molab.marimo.io/notebooks), choose **+ New notebook → Mirror from GitHub**, paste the teacher's verified notebook URL, fork your copy, select a GPU in **Configure compute**, then **Save and restart → Run all**. The [Korean README](avllm_interpretability/README_kr.md) follows priorities **1, 2, 3, 4, 7, 8**: answer→audio/video, original/silent controls, same-language prompt tasks, layer bands, new clips, and advanced paths. Start with four teacher-forced runs (original/silent × audio/video), then progress to the other activities and the diversity probe. Both student forms keep **8 frames fixed**; language variation is only an optional note in the prompt cell.

The final classroom code (`936e9f4`) passed **316 CPU tests and both notebook marimo checks**. Molab RTX Pro 6000 QA produced **12 experiments (9 teacher-forced, 3 diversity)** on source `914e56b`; the final export cell was then tested in that live notebook against all 12 records, including reconstruction and SHA-256 verification of the large JSON and Markdown copy fallbacks. This was not a fresh startup of the complete final source. [QA details and remaining checks](avllm_interpretability/QA_IMPROVEMENTS.md) distinguish this scope from the earlier `61d540d` restart-recovery tests. New-clip upload was blocked by the test browser's file-access permission; direct download receipt, whole-class GPU allocation, server recreation, and a final-source restart remain unverified. The English notebook retains its earlier setup and teaching flow.

Local execution requires [uv](https://docs.astral.sh/uv/), **Python ≥ 3.11**, and a compatible NVIDIA GPU. The Qwen2.5-Omni-3B weights download from Hugging Face on first run. GPU and host-memory requirements depend on the input and runtime; the measurements below are a reference run, not a minimum-memory guarantee.

*Measured on an RTX 3090:* the model is 4.70 B parameters (bf16, talker freed) and occupies **8.88 GiB**; a full guided run peaks at **13.70 GiB**. It used to load a second SDPA copy for the logit lens, which peaked at 22.08 GiB and ran out of memory on a 24 GB card — so if you are working from an older checkout and see `CUDA out of memory`, that is why.

> **No GPU?** Set `USE_PRECOMPUTED = True` near the top to read saved guided captions, probes, and summarized attention without model weights or GPU inference. New layer-band generation, playground submissions, and teacher-forced measurements need a GPU. Replay supports interpretation and experiment design; dependency installation and repository access are still needed, and it cannot test new parameters.

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
python -m pytest feedback-atlas           # feedback map: server, projection, privacy
```

`feedback-atlas` has its own environment (`uv venv --python 3.12`) — numpy 2.5 and
scipy 1.18 both require 3.12, so it cannot share the notebooks' 3.11. Its suite
follows the same rule as the rest: CPU only, no weights, no network, with heavy
dependencies skipping rather than failing.

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

The Korean AVLLM [instructor guide](avllm_interpretability/CLASSROOM_GUIDE_kr.md),
[student worksheet](avllm_interpretability/WORKSHEET_kr.md), and
[setup guide](avllm_interpretability/README_kr.md) are available. They cover the
updated Korean notebook; the broader WP-6 bilingual package remains planned in
the [PRD](CTP49906_W7-11_PRD.md). Its unchecked curriculum requirements are not
certified by the notebook QA.

[`feedback-atlas/`](feedback-atlas/) is a *neighbour* of the blinded
audience-response surface that PRD also anticipates, and deliberately not the same
thing. It collects opinions during a presentation and maps them; the blinded
surface in `.omx/plans/prd-avllm-arts-integrated-classroom.md` (FR-6, §7.4)
specifies a different data contract — blindness attestation, permission scope,
withdrawal — and a different release path, a marimo notebook under `audience/`.
The two overlap on `reviewer_id`/`target_id` and on post-semester
de-identification, and should be reconciled before either is used for research
rather than for teaching. The English research/design worksheet is
[`avllm_interpretability/WORKSHEET.md`](avllm_interpretability/WORKSHEET.md); the
English notebook exports its ledger in that format. The Korean notebook also
exports complete recorded settings, captions, and token results as JSON, with
Markdown and a visible copy fallback. Verify the exported files on the student
device before ending the session.
