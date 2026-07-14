# AVLLM Interpretability

This workspace contains the `avllm_interpretability` experiments for Qwen2.5 Omni:

- Logit Lens analysis
- Attention Knockout analysis

The commands below use the included video, `avllm_interpretability/assets/02321.mp4`.

## Code sources

- [Anthropic Jacobian Lens](https://github.com/anthropics/jacobian-lens)
- [Ramaneswaran AVLLM Interpretability](https://github.com/ramaneswaran/avllm_interpretability)

## Prerequisites

- Linux or macOS with Bash, or Windows with PowerShell and WSL/Git Bash
- Git
- An NVIDIA GPU and a CUDA-compatible driver are strongly recommended; the environment installs the CUDA 12.4 PyTorch build
- Enough disk space for the Qwen2.5-Omni-3B model, which is downloaded from Hugging Face on the first run
- At least 24 GB of VRAM for the included defaults; the scripts sample the included video at 8 frames to fit a 24 GB RTX 3090

## Install uv

Check whether uv is already available:

```bash
uv --version
```

If that command is not found, install uv on Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

If the `source` command cannot find the uv environment file, open a new terminal and run `uv --version` again. On Windows PowerShell, use:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Create and install the project environment

From this workspace root, enter the experiment directory:

```bash
cd avllm_interpretability
```

Create a Python 3.10 virtual environment with pip seeded into it, then install the pinned dependencies:

```bash
uv venv --python 3.10 --seed .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The requirements file includes the PyTorch CUDA 12.4 index and the matching `torchvision` package. It also uses the published `transformers==4.52.0` release because the historical `4.52.0.dev0` build is no longer available.

Activate the environment before running an experiment:

```bash
source .venv/bin/activate
which python
python --version
uv pip check
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

You can also run commands without activation by replacing `python` with `.venv/bin/python` on Linux/macOS or `.venv\Scripts\python.exe` on Windows.

## Run the notebook

From `avllm_interpretability`, install the notebook tools once and start JupyterLab:

```bash
uv pip install --python .venv/bin/python jupyterlab ipykernel matplotlib
.venv/bin/jupyter lab
```

Open [CTP49906_avllm.ipynb](avllm_interpretability/CTP49906_avllm.ipynb), select the **AVLLM Interpretability (.venv)** kernel, update `VIDEO_PATH` in the parameter cell for your video, then use **Run All**. The notebook runs both experiments and saves visualized results under `notebook_results/`.

## Run Logit Lens

From `avllm_interpretability`, with `.venv` activated:

```bash
python src/logitlens_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```

This asks the model to describe what it hears, captures hidden states from the thinker layers, and writes `logit_lens_audio_token_analysis.csv` in the current directory. The script also prints a generated-text sanity check.

## Run Attention Knockout

The default experiment blocks generated-to-video attention for layers 0 through 35:

```bash
python src/attention_knockout_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```

The script prints the video path, knockout rules, token counts, and generated text. The current entrypoint defines a `save_knockout_data` helper but does not call it, so it does not write the normally planned `generated2videoL0_35_single_video.json` file by default.

To choose a different rule or prompt, pass the optional arguments. Rules are semicolon-separated and use `source,target,start_layer,end_layer`:

```bash
python src/attention_knockout_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4 \
  --knockout-rules "audio,video,0,35" \
  --query_text "Describe what you see and hear"
```

## Troubleshooting

### `uv: command not found`

Open a new terminal after installing uv, or source its environment file:

```bash
source "$HOME/.local/bin/env"
```

### `ModuleNotFoundError` or the wrong Python is used

Make sure the shell is inside `avllm_interpretability` and the environment is active:

```bash
cd avllm_interpretability
source .venv/bin/activate
which python
```

The reported Python path should end in `avllm_interpretability/.venv/bin/python`.

### CUDA is unavailable

Verify the driver and GPU visibility with `nvidia-smi`. The installed Torch package is CUDA-enabled, but it cannot use a GPU when the host driver or container does not expose one. These experiments are large and may be impractical on CPU-only machines.

### Model download or memory errors

The first run downloads model files from Hugging Face. Confirm network access, available disk space, and sufficient system RAM/VRAM. If the model is already cached elsewhere, pass that local model directory to `--model_path` instead of `Qwen/Qwen2.5-Omni-3B`.
