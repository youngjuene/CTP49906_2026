# Do Audio-Visual Large Language Models Really See and Hear?

<p align="center">
  <a href="https://arxiv.org/abs/2604.02605">
    <img src="https://img.shields.io/badge/arXiv-2604.02605-b31b1b" alt="arXiv"/>
  </a>
  <a href="https://avllm-interpretability.github.io">
    <img src="https://img.shields.io/badge/Project-Website-blue" alt="Website"/>
  </a>
</p>

<p align="center">
  <b>English</b> · <a href="README.ko.md">한국어</a>
</p>

Code for experiments conducted in the paper, with Qwen 2.5 Omni as the representative model.


## Installation
```bash
pip3 install -r requirements.txt
```

## Experiments

### Logit Lens Experiment
```bash
python src/logitlens_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```

### Attention Knockout Experiment
```bash
python src/attention_knockout_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```

## Classroom notebook: `CTP49906_avllm_molab.py`

`CTP49906_avllm_molab.py` is a self-contained [marimo](https://marimo.io) notebook that
turns the two experiments above into a **teaching walkthrough plus a live playground**.
It is meant to be opened in [molab](https://marimo.io/molab) (a free hosted GPU runtime),
read top-to-bottom once so the mechanics are clear, and then *poked at* — swap the clip,
the prompt, the number of frames, and above all the **attention knockouts** — to build
intuition for the paper's central question: **when an audio-visual LLM answers, is it
really using both what it sees and what it hears?**

### Why it exists

A caption like *"a person is playing the piano"* looks equally correct whether the model
watched the video, listened to the audio, or leaned on language priors. This notebook gives
students two interpretability probes for forming and testing hypotheses about those cases:

- **Logit Lens** — applies `lm_head` directly to each layer's raw residual-stream state at
  **audio-token positions**. This probe omits the thinker's final RMSNorm, and audio
  positions do not have a calibrated next-token LM objective. Its decoded tokens are
  therefore diagnostics, not literal intermediate next-token predictions.
- **Attention Knockout** — blocks selected direct source→target attention edges in selected
  layers (e.g. generated-token queries reading video-token keys) and re-runs. It does **not**
  remove a modality or erase indirect/residual routes, so conclusions are conditional on the
  particular edge intervention.

The goal is not to reproduce a number; it's to develop a feel for *reading a model from the
inside* and forming falsifiable "if I block X, the output should change like Y" hypotheses.

### Running it (molab)

1. Open the notebook in molab and **attach a GPU** via the notebook-specs button in the
   header (it uses `cuda:0`; the 3B thinker fits comfortably).
2. Run the cells top-to-bottom. The setup cell pip-installs dependencies into the kernel,
   restores `torchvision.io.read_video` with a small PyAV shim (recent torchvision dropped
   the video decoder), and clones this repo (`src/` + the sample clip) so the experiment
   code is importable.
3. First run downloads the model weights (~8 GB) and is the slow one; later cells reuse the
   loaded model.
4. The Qwen weights are pinned to the immutable revision recorded in the notebook and
   `precomputed/meta.json`. Before the class release, set `REPO_REF` to the published course
   tag too; when opened from a local checkout, the notebook uses that checkout as-is.

You can also run it locally with `uvx marimo edit CTP49906_avllm_molab.py` on a CUDA
machine — the `# /// script` header pins compatible dependency versions.

**No GPU? GPU-free replay.** If molab's GPU is unavailable, set `USE_PRECOMPUTED = True`
in the config cell near the top: every non-interactive guided-demo panel then renders from
committed artifacts in [`precomputed/`](precomputed/) with no GPU and no 8 GB download
(each is labelled *"Saved course replay — no GPU"*). It defaults to `False` (live model); the
interactive playground and teacher-forcing sections still need a GPU and fail loudly if
run in this mode. Regenerate the artifacts on a CUDA box with
`python scripts/generate_precompute.py`.

### A tour of the cells (the read-through)

| Cell | What it does | What to notice |
| --- | --- | --- |
| **Setup** | Installs deps, patches the video reader, clones the repo. | Nothing to tune; just let it finish. |
| **Guided-demo reference** | Central knobs: `VIDEO_PATH`, `NFRAMES`, `LOGIT_PROMPT`, `ATTENTION_PROMPT`, `KNOCKOUT_RULES`, `MAX_NEW_TOKENS`. | Leave unchanged first; edit only to redesign the shared reference run. |
| **Video preview** | Plays the exact clip (frames **and** embedded audio) sent to Qwen. | Whatever the model can't perceive here, it can't answer from. |
| **Model + helpers** | Loads Qwen2.5-Omni-3B (talker freed — this only needs the *thinker*) and builds the token-type map. | Prints `token counts:` — how many `audio` / `video` / `query_text` tokens your prompt produced. |
| **Logit Lens** | One forward pass; decodes per-layer predictions at audio positions to a CSV; also prints the caption. | The caption is the model's "final answer" for comparison. |
| **Diversity by layer** | Two plots: how many *distinct* probe tokens each layer decodes at audio positions, and how dominant the top token is. | A descriptive argmax statistic—not uncertainty, quality, or proof of fusion. |
| **Attention Knockout** | Generates a **baseline** caption and a **knockout** caption side-by-side using `KNOCKOUT_RULES`. | The whole point: does the answer *change* when a pathway is cut? |
| **Captured attention** | Baseline and knockout heatmaps plus their delta, showing final-query attention mass by token type. **Descriptive, not causal importance.** | Masking mechanically redistributes attention; read this *alongside* outcome measures. |
| **Teacher-forced Δ log-lik** | Feeds the baseline caption back in tagged `answer` and scores its per-token log-probability change under the same direct-edge knockout. | Continuous and deterministic; reports additive total and length-normalized mean. |
| **🎛️ Playground** | Interactive form (below) that re-runs the logit-lens diversity measurement on your choices. | Where students spend most of their time. |
| **🎯 Teacher forcing** | Interactive form for the Δ log-lik measurement: your clip, prompt, target modality, and layer band, with the source fixed to `answer`. | Where `answer → audio` — inert everywhere else — becomes a real experiment. |
| **Synthesis challenge** | Audits the evidence, then asks for a new modality-routing or fusion design. | Separate observation from mechanism and state what would falsify the design. |

### Reading a knockout rule

Every knockout is a tuple **`(source, target, start_layer, end_layer)`** and means:

> *In thinker layers `[start_layer, end_layer)`, forbid tokens of type `source` from
> attending to tokens of type `target`.*

Two things students trip on, worth stating up front:

- **It is directional.** Attention flows *from* the source (the token doing the looking)
  *to* the target (the token being read). `generated → video` (the answer can't see the
  frames) is a completely different intervention from `video → generated`. The meaningful
  direction is almost always *a later token reading an earlier one*, because the model is
  causally masked — a token can only attend to itself and everything before it.
- **`end_layer` is exclusive.** `[0, 36)` means layers 0 through 35, i.e. all of them for
  the 3B thinker (the notebook prints the true layer count). Narrowing the window tests
  **layer-band sensitivity**. It does not by itself localize a fusion module: information
  may already have travelled indirectly or be rerouted through unblocked edges.

The six intervention token types (only some are modalities) used to tag positions:

| Type | What it is |
| --- | --- |
| `query_text` | The words of your prompt / instruction. |
| `audio` | Tokens from the video's **soundtrack**. |
| `video` | Tokens from the sampled **frames**. |
| `image` | Still-image tokens — absent for a video clip, so inert here. |
| `generated` | Tokens the model **produces** during autoregressive decoding (exist only while generating). |
| `answer` | The model's own caption **teacher-forced back in** as input, so a single forward pass can score it. Used by the teacher-forced Δ log-lik section; tagged by position, not by token id. |

### A catalog of knockout pairs (what each one is asking)

These are the interventions worth trying. The first three use `generated` as the source and
so belong in the **Attention Knockout cell** (they act during generation); the last two
reshape the **audio-position logit lens** and belong in the **playground**.

| Rule | The question it poses | What a change in the output means |
| --- | --- | --- |
| `generated → video` | *What changes when generated-token queries cannot directly read video-token keys in this layer band?* | A change supports sensitivity to those edges; compare matched controls before attributing visual grounding. No change can reflect redundancy or earlier indirect transfer. |
| `generated → audio` | *What changes when generated-token queries cannot directly read audio-token keys?* | A change supports sensitivity to those edges, not by itself “genuine listening”; test silent/mismatched controls and rival routes. |
| `generated → query_text` | *What changes when generated tokens cannot directly re-read the instruction?* | Drift is consistent with reliance on those edges, while an unchanged answer may reflect instruction information already encoded elsewhere. |
| `audio → video` | *Are audio-position probe statistics sensitive to direct video-key edges?* | A diversity shift is an effect to explain; it does not prove visual→audio fusion or semantic improvement. |
| `video → audio` | *Are video-token queries sensitive to direct audio-key edges?* | A later change motivates a cross-modal hypothesis, which still needs controls that distinguish it from masking/renormalization effects. |

> **Why `generated` does nothing in the diversity scoreboard.** The diversity scoreboard runs a
> single *forward pass* over the prompt (no autoregressive decoding), so there are **no
> `generated` tokens** for a rule to act on. A `generated → …` rule there blocks nothing and the
> Δ is flat — the notebook warns you when you try. To move the audio-position score, make the
> **source** a modality that is actually present: `audio`, `video`, or `query_text`.
>
> **Where the answer *can* be the source: teacher forcing.** The most intuitive question —
> *"if the answer can't hear the soundtrack, does it still describe the sound?"* — needs the
> answer to exist as input. The **teacher-forced Δ log-likelihood** section does exactly that:
> it generates the caption once, feeds it back in tagged **`answer`**, and scores
> `answer → audio` (or `→ video`, `→ query_text`) as a continuous, deterministic **Δ
> log-lik = knockout − baseline** — negative meaning the model believes its own caption *less*
> once those direct edges are cut. This is a controlled intervention metric, not proof that
> the target modality supplied the caption's semantics; `answer` is meaningful here because
> it exists as an input token type.

### The research playground (the tweak-it part)

The `🎛️` section wraps the diversity measurement in a form — **nothing runs until you
press ▶** — and reuses the already-loaded model, so iterations are fast and need no extra
VRAM. Controls:

- **Clip** — explicitly choose **Default**, matched **Silent control**, or **Upload** in both
  forms. Upload requires a file; it never silently falls back. Uploaded bytes are stored and
  cached by SHA-256 under a path-safe generated filename. Preflight rejects files over 250 MB
  or 120 seconds, over 1080p/60 FPS or the decoded-memory budget, unknown-duration media,
  missing audio/video streams, and clips with no decodable video frame.
- **Frames** — 2–32; more frames = richer visual context (and slower).
- **Prompt** — the instruction; try steering it toward sound vs. sight.
- **Knockout on/off**, then either the **single-rule** dropdowns (source, target, layer
  range) or the **advanced** field for several rules as `source,target,start,end` separated
  by `;` (e.g. `audio,video,0,36 ; audio,image,0,36`). The advanced field wins when filled.
- **Compare** — also run a no-knockout baseline so the scoreboard can show a per-layer
  **Δ (knockout − baseline)**.

Reading the scoreboard: each layer is scored by **how many distinct argmax probe tokens it
decodes across the audio positions**. A **negative Δ** means the knockout produced fewer
distinct probe tokens. It does not establish that representations improved, degraded, or
fused modalities. A flat Δ can mean no sensitivity, redundant/indirect routes, an insensitive
metric, or (for a `generated` source) an intervention that could not act.

### Suggested experiments for students

Log every run in the [lab worksheet](WORKSHEET.md) — hypothesis **before** ▶, result,
verdict. Its final block is the same effect-and-control question the final project is
graded on.

1. **Sight vs. sound, same clip.** Run `generated → video` and then `generated → audio` in
   the knockout cell with the *"see and hear"* prompt. Which knockout changes the caption
   more? What does that establish about sensitivity to those direct edge sets—and what
   additional control would be needed before claiming modality reliance?
2. **Test a candidate cross-modal band.** In the playground, run `audio → video` over
   `[0, 12)`, then `[12, 24)`, then `[24, 36)`. Which band changes diversity most? What
   control or alternative edge rule would falsify the claim that this reflects fusion?
3. **Prompt steering.** Keep the clip fixed and switch the prompt between *"what you hear"*
   and *"what you see"*. Do the audio positions decode differently even before any knockout?
4. **Bring your own clip.** Upload a video where audio and vision *disagree* (e.g. narration
   over unrelated footage) and see which modality the model reports.
5. **Stack rules.** Use the advanced field to knock out `audio → video` **and**
   `audio → query_text` at once — does starving the audio stream of *both* neighbors compound
   the diversity change, cancel it, or reverse it?
6. **Sight vs. sound, quantified.** In the 🎯 teacher-forcing section, run `answer → audio`
   and then `answer → video` on the same clip and prompt. Experiment 1 asked which knockout
   *changes the caption more*; this asks for total and **mean per-token** Δ log-likelihood.
   Do the binary diff and continuous measurement agree on edge sensitivity, and what still
   prevents a stronger modality-reliance claim?
7. **Which layer band is this score sensitive to?** Run `answer → audio` over `[0, 12)`,
   then `[12, 24)`, then `[24, 36)`. Which layer band, when cut, costs the caption the most
   log-likelihood? Compare with experiment 2, then propose an explanation that does not
   assume a localized fusion module.
8. **A null that means something.** Run `answer → audio` on `assets/02321_silent.mp4` — same
   frames as the sample clip, but the soundtrack is digital silence. The audio tokens still
   exist: predict whether Δ should approach zero, then test it. Silence is not a guaranteed
   null because preprocessing, positions, and attention renormalization remain. If the two
   clips behave similarly, which interpretation is falsified—and what control comes next?

## Citation

```bibtex
@misc{selvakumar2026audiovisuallargelanguagemodels,
      title={Do Audio-Visual Large Language Models Really See and Hear?},
      author={Ramaneswaran Selvakumar and Kaousheik Jayakumar and S Sakshi and Sreyan Ghosh and Ruohan Gao and Dinesh Manocha},
      year={2026},
      eprint={2604.02605},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.02605},
}
```

## WP-6 classroom release notes

Use the bilingual [`study_materials/wp6/`](../study_materials/wp6/) runbook and
student quick-start alongside this notebook. The replay manifest pins the model
revision and artifact checksums; the audience response surface is
[`../audience/CTP49906_audience_response_molab.py`](../audience/CTP49906_audience_response_molab.py).
This is a teaching-only candidate release until the listed human accessibility,
localization, licensing, and governance gates are reviewed.
