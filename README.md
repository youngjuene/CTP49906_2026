# CTP49906 Interpretability Labs

Classroom interpretability labs for Qwen2.5-Omni and Qwen3.5: **Logit Lens**, **Attention Knockout**, teacher-forced intervention scoring, and the **Jacobian Lens**. The marimo scripts pair a fixed guided run with a hypothesis-led playground rather than ending at a canned reproduction.

## Contents

- [`avllm_interpretability/`](avllm_interpretability/) — the experiments ([README](avllm_interpretability/README.md)), adapted from [ramaneswaran/avllm_interpretability](https://github.com/ramaneswaran/avllm_interpretability) ([project page](https://ramaneswaran.github.io/avllm_interpretability/))
- [`jacobian-lens/`](jacobian-lens/) — vendored [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) reference code ([paper](https://transformer-circuits.pub/2026/workspace/index.html))

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with ≥24 GB VRAM. The Qwen2.5-Omni-3B weights download from Hugging Face on first run.

```bash
cd avllm_interpretability
uv venv --python 3.10 --seed .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate

# Logit Lens → logit_lens_audio_token_analysis.csv
python src/logitlens_experiment.py --model_path Qwen/Qwen2.5-Omni-3B --video_path assets/02321.mp4

# Attention Knockout (rules: source,target,start_layer,end_layer)
python src/attention_knockout_experiment.py --model_path Qwen/Qwen2.5-Omni-3B --video_path assets/02321.mp4
```

Or run both from the classroom marimo notebook [`CTP49906_avllm_molab.py`](avllm_interpretability/CTP49906_avllm_molab.py) — open it in [molab](https://marimo.io/molab) with a GPU attached, or locally with `uvx marimo edit`. Its guidebook (cell tour, knockout catalog, suggested experiments) is in [`avllm_interpretability/README.md`](avllm_interpretability/README.md).

The companion [`CTP49906_jlens_molab.py`](jacobian-lens/CTP49906_jlens_molab.py) first compares the course reference Qwen3.5-4B Jacobian lens with the vanilla logit lens, then exposes prompt offset, layers, top-k, slice, filtering, and lens-estimator choices in a submit-gated research playground before an architecture-transfer synthesis. See the [`jacobian-lens` classroom guide](jacobian-lens/README.md#classroom-marimo-demo).

## Classroom release guidance

The WP-6 bilingual instructor, student, privacy, troubleshooting, and worksheet
migration materials are in [`study_materials/wp6/`](study_materials/wp6/). The
blinded audience response surface is [`audience/CTP49906_audience_response_molab.py`](audience/CTP49906_audience_response_molab.py);
its saved replay and accessibility alternatives are checksum-bound in the WP-6
release manifest. These materials describe a teaching-only candidate release;
human accessibility, localization, licensing, and research-governance review
remain open.
