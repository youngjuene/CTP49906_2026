# AVLLM Interpretability

Reproducible interpretability experiments for the audio-visual LLM Qwen2.5-Omni: **Logit Lens** and **Attention Knockout**. Commands below use the bundled sample video `avllm_interpretability/assets/02321.mp4`.

## Contents

- [`avllm_interpretability/`](avllm_interpretability/) — the experiments ([README](avllm_interpretability/README.md)), adapted from [ramaneswaran/avllm_interpretability](https://github.com/ramaneswaran/avllm_interpretability)
- [`jacobian-lens/`](jacobian-lens/) — vendored [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) reference code

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

Or run both from [`CTP49906_avllm.ipynb`](avllm_interpretability/CTP49906_avllm.ipynb) — select the **AVLLM Interpretability (.venv)** kernel, set `VIDEO_PATH`, then **Run All**.
