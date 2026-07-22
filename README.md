# CTP49906 audiovisual interpretability studio

<p align="center"><b>English</b> · <a href="README.ko.md">한국어</a></p>

This repository contains an arts-integrated classroom route for making,
examining, comparing, and revising an audiovisual work. The primary treatment
notebook is **Counterpoint Lens**; a separate no-GPU audience surface preserves
blindness during human interpretation. Neutral production, session, audience,
and export contracts live in `curriculum_common/`.

## Classroom entry points

- [Counterpoint Lens notebook](avllm_interpretability/CTP49906_avllm_molab.py)
  — guided saved replay plus optional live interpretability investigation.
- [Blinded audience response](audience/CTP49906_audience_response_molab.py)
  — validates an allowlisted presentation packet and downloads one independent
  audience reading without a GPU or model.
- [English/Korean classroom guidance](study_materials/wp6/README.md) —
  instructor runbook, student quick-start, worksheet migration, privacy/data
  dictionary, troubleshooting, replay identity, and visual alternatives.
- [AVLLM technical and classroom guide](avllm_interpretability/README.md).
- [Jacobian Lens companion](jacobian-lens/README.md#classroom-marimo-demo).

## Safe quick start

The required route begins with **Saved course replay**. It uses checked-in
outputs and does not allocate a GPU or download model weights. Open the treatment
notebook locally with:

```bash
uvx --python 3.10 marimo@0.23.14 edit \
  avllm_interpretability/CTP49906_avllm_molab.py
```

Open the blinded audience surface separately:

```bash
uvx --python 3.10 marimo@0.23.14 run \
  audience/CTP49906_audience_response_molab.py
```

Teaching mode is the default and fail-safe mode. The notebooks do not
automatically send student artifacts, process records, or audience responses to
an instructor. A local private download or permission-valid blinded packet is
created only after the user activates its control.

## Replay and release status

[The replay manifest](study_materials/wp6/replay_manifest.json) publishes the
immutable model revision, coded stimulus checksum, artifact hashes, and exact
GPU generation command. The committed pack is a candidate teaching replay; this
CPU-only change verifies its bytes but does not claim to have regenerated the
GPU outputs.

The software and materials are **not research-ready**. Institutional governance,
licensed-stimulus release, instrument quality, comparison fidelity, audience
validity, accessibility/localization review, immutable course tagging, and pilot
thresholds remain accountable human gates.

## Development checks

```bash
python3.10 -m py_compile audience/*.py curriculum_common/*.py
uvx --python 3.10 marimo@0.23.14 check --strict \
  audience/CTP49906_audience_response_molab.py
PYTHONPATH=. uvx --python 3.10 pytest -q tests
uvx --python 3.10 ruff check audience curriculum_common tests
```

The live AVLLM path has additional CUDA/model dependencies documented in
[avllm_interpretability/README.md](avllm_interpretability/README.md).
