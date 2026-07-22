# Counterpoint Lens AVLLM classroom notebook

<p align="center"><b>English</b> · <a href="README.ko.md">한국어</a></p>

`CTP49906_avllm_molab.py` is a Marimo classroom studio built around the
Qwen2.5-Omni interpretability code from *Do Audio-Visual Large Language Models
Really See and Hear?* It combines an audiovisual production/revision route with
bounded logit-probe, direct attention-edge intervention, and teacher-forced
answer-distribution observations.

The notebook asks what available evidence suggests—not whether a plausible
caption proves that a model “really” saw or heard something.

## Required classroom route

The rendered notebook has four sections and eight stages:

1. **Prepare your project**
   - Orient — make, test, revise
   - Compose — plan your first cut
   - Register — first cut (V1)
2. **Guided demonstration**
   - Observe — shared reference
3. **Exploratory playground**
   - Experiment — change one thing at a time
   - Compare — three readings of one work
   - Revise — explanation and second cut (V2)
4. **Synthesis and architecture challenge**
   - Synthesize — portfolio and bounded proposal

Required, choice, and advanced activities are visibly separated. Forms snapshot
inputs on submission; editing a draft does not commit a run or reflection.

## Start with saved replay

Choose **Saved course replay** in the first notebook control. It uses the
checksum-bound files in `precomputed/`, needs no GPU, and does not download
model weights. Live-only actions remain unavailable with an explanation.

The immutable model/stimulus identity, artifact checksums, generation command,
and candidate-release limitations are published in
[`../study_materials/wp6/replay_manifest.json`](../study_materials/wp6/replay_manifest.json).
The current CPU verification checks the committed bytes; it does not claim a new
GPU regeneration.

Open locally:

```bash
uvx --python 3.10 marimo@0.23.14 edit CTP49906_avllm_molab.py
```

The live route uses the pinned dependencies in the script header and requires a
CUDA runtime with the instructor-approved resource envelope. The first live
model load downloads Qwen2.5-Omni weights. Do not make live execution a
requirement when the release environment has not been rehearsed.

## Reading the evidence

- **Raw probe score dispersion** is a compact layer/position diagnostic, not
  calibrated uncertainty or an intermediate next-token claim.
- **Captured attention mass** is descriptive. Masking mechanically
  redistributes attention and does not make attention a causal explanation.
- **Direct attention-edge knockout** blocks selected source-query to target-key
  edges for a layer range. It is not modality removal.
- **Teacher-forced answer-distribution change** scores a fixed answer under a
  controlled direct-edge intervention. It is not free-generation uncertainty.
- Position comparisons require compatible token-layout fingerprints. The
  notebook blocks or downgrades incompatible comparisons.

Every interpretation should identify an observation, a bounded inference, a
limitation, a rival explanation, and the next control.

## Blinded audience comparison

Creators export an allowlisted presentation packet. Respondents use the separate
no-GPU [audience surface](../audience/CTP49906_audience_response_molab.py).
Creator intention, condition, private identifiers, and model output remain
absent before reveal.

The creator notebook accepts two distinct schema-valid blinded readings for the
same exchange before revealing creator/model readings. The comparison records
overlap, divergence, and an unrepresented reading; it never treats agreement as
truth or two respondents as a defensible research sample.

## Data, access, and recovery

Teaching mode is the default. The notebook does not automatically send student
artifacts, process records, or audience readings to an instructor. User-initiated
private downloads and permission-valid audience packets are separate egress
classes. Research export remains unavailable without valid approved
configuration and independent permissions.

Captions, transcripts, descriptions, keyboard checks, instructor/student routes,
privacy fields, troubleshooting, and bilingual companions are documented in
[the classroom guidance pack](../study_materials/wp6/README.md). These supports
are human-correctable and never silently become model input.

The legacy [WORKSHEET.md](WORKSHEET.md) remains a teaching note. New work uses
the notebook's versioned Prepare → Run → Reflect records; see the
[worksheet migration note](../study_materials/wp6/worksheet_migration.en.md).

## Live experiment commands

The original command-line experiments remain available after installing
`requirements.txt`:

```bash
python src/logitlens_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4

python src/attention_knockout_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```

## Release status

The repository provides candidate teaching materials, not a research-ready
release. Accessibility/localization, licensed stimuli, audience validity,
instrument quality, ethics/data governance, comparison fidelity, immutable
course tagging, and pilot thresholds still require named human review.

## Citation

```bibtex
@misc{selvakumar2026audiovisuallargelanguagemodels,
  title={Do Audio-Visual Large Language Models Really See and Hear?},
  author={Ramaneswaran Selvakumar and Kaousheik Jayakumar and S Sakshi and Sreyan Ghosh and Ruohan Gao and Dinesh Manocha},
  year={2026},
  eprint={2604.02605},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2604.02605}
}
```
