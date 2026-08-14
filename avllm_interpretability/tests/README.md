# Running the tests

These are CPU-only logic tests: no GPU, no model weights, no network. They cover
the pure functions behind the notebook's experiments (attention knockout,
teacher forcing, precomputed replay, run ledger, rule reach).

## The command

```
/home/june/anaconda3/envs/sent/bin/python -m venv --system-site-packages /path/to/pytest311   # once
/path/to/pytest311/bin/python -m pip install pytest transformers==4.52.0 qwen-omni-utils==0.0.9 accelerate==1.14.0
/path/to/pytest311/bin/python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu124 torchvision==0.21.0+cu124

/path/to/pytest311/bin/python -m pytest avllm_interpretability
```

`pytest.ini` sets `testpaths = tests`, so from inside `avllm_interpretability/`
a bare `python -m pytest` is enough. `tests/conftest.py` puts
`avllm_interpretability/` on `sys.path`, so `from src.x import y` resolves no
matter which directory you invoke pytest from.

## Which interpreter

`~/anaconda3/envs/sent/bin/python` — Python 3.11.11 with torch 2.6.0+cu124 and
matplotlib 3.10.1, matching `requirements.txt`. That env has no pytest and is
not ours to modify, so the command above layers a `--system-site-packages` venv
on top of it: torch and matplotlib come from the conda env, pytest and the four
notebook-side packages are installed into the venv only.

Put that venv **outside the repo** (a scratch/tmp path). The repo's `.gitignore`
lists only `.omx/` — it ignores neither `.venv` nor `__pycache__` — so an env
created inside the working tree would show up in `git status`.

Do not use the committed `.venv` directories (`avllm_interpretability/.venv`,
`jacobian-lens/.venv`). They are Python 3.10.16; see below.

## Why >= 3.11

The notebook's `_ensure_packages` installs `wigglystuff==0.5.21` with
`check=True`, and every wigglystuff >= 0.5.15 declares `Requires-Python >=3.11`.
Under the committed 3.10.16 venv pip refuses the pin outright —

```
ERROR: Ignored the following versions that require a different python version:
       ... 0.5.21 Requires-Python >=3.11 ...
ERROR: No matching distribution found for wigglystuff==0.5.21
```

— so the notebook cannot start in its own committed venv. Testing on 3.11 keeps
the suite on the same interpreter floor as the thing it is testing. (The two
oldest test files happen to pass under 3.10 as well, since they touch no widget
code; anything importing the widget-backed modules will not.)

## Markers

`needs_anywidget` — for tests that need the optional `anywidget` dependency.
`tests/conftest.py` skips them automatically when it is missing, so the suite
stays green on a bare interpreter. The marker assumes the test *module* still
imports without anywidget: keep widget classes out of module-level imports and
test the pure prep functions directly.

**Install `anywidget` in the test environment.** Two tests in
`test_probe_grid.py` carry this marker, and they are the only place the
`ProbeGrid` class is ever constructed — they pin the pack-to-trait mapping, that
the two `Bytes` traits leave as out-of-band buffers, and that the ESM never
writes a trait back. That last one is the invariant keeping a widget interaction
from reaching the GPU, so skipping it is not a neutral default:

```bash
python -m pip install 'anywidget>=0.9.2'
```

Without it you get `13 passed, 2 skipped` and the widget contract is unverified.

`--strict-markers` is on, so a misspelled marker is a collection error rather
than a test that silently never runs.

## Notebook graph check

The tests do not exercise the marimo notebook. To check that its dataflow graph
is still acyclic and every cell parses:

```
uv run --no-project --python 3.12 --with marimo==0.23.14 marimo export script <nb> -o /tmp/out.py
```

with `<nb>` = `CTP49906_avllm_molab.py`. A non-zero exit means a cycle or a
multiple-definition error.
