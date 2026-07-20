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
students two interpretability tools to tell those cases apart:

- **Logit Lens** — decodes the model's *intermediate* prediction at each layer, read off
  the **audio-token positions**, so you can watch when (and whether) audio content becomes
  legible as you climb the network.
- **Attention Knockout** — surgically cuts one information pathway (e.g. *the answer may no
  longer look at the video frames*) and re-runs, so you can see, causally, what breaks when
  a modality is taken away.

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

You can also run it locally with `uvx marimo edit CTP49906_avllm_molab.py` on a CUDA
machine — the `# /// script` header pins compatible dependency versions.

**No GPU? GPU-free replay.** If molab's GPU is unavailable, set `USE_PRECOMPUTED = True`
in the config cell near the top: every non-interactive **W7-W9** plot then renders from
committed artifacts in [`precomputed/`](precomputed/) with no GPU and no 8 GB download
(each is labelled *"replayed from cache"*). It defaults to `False` (live model); the
interactive playground and teacher-forcing sections still need a GPU and fail loudly if
run in this mode. Regenerate the artifacts on a CUDA box with
`python scripts/generate_precompute.py`.

### A tour of the cells (the read-through)

| Cell | What it does | What to notice |
| --- | --- | --- |
| **Setup** | Installs deps, patches the video reader, clones the repo. | Nothing to tune; just let it finish. |
| **Parameters** | Central knobs: `VIDEO_PATH`, `NFRAMES`, `LOGIT_PROMPT`, `ATTENTION_PROMPT`, `KNOCKOUT_RULES`, `MAX_NEW_TOKENS`. | This is the one cell you edit to change the *fixed* run. |
| **Video preview** | Plays the exact clip (frames **and** embedded audio) sent to Qwen. | Whatever the model can't perceive here, it can't answer from. |
| **Model + helpers** | Loads Qwen2.5-Omni-3B (talker freed — this only needs the *thinker*) and builds the token-type map. | Prints `token counts:` — how many `audio` / `video` / `query_text` tokens your prompt produced. |
| **Logit Lens** | One forward pass; decodes per-layer predictions at audio positions to a CSV; also prints the caption. | The caption is the model's "final answer" for comparison. |
| **Diversity by layer** | Two plots: how many *distinct* tokens each layer decodes at audio positions, and how dominant the top prediction is. | A low-diversity layer is "committed"; a high-diversity layer is still "deciding". |
| **Attention Knockout** | Generates a **baseline** caption and a **knockout** caption side-by-side using `KNOCKOUT_RULES`. | The whole point: does the answer *change* when a pathway is cut? |
| **Captured attention** | Heatmap of how much the final query attends to each modality, per captured layer. **Descriptive, not causal.** | Read it *alongside* the text diff, not instead of it. |
| **🎛️ Playground** | Interactive form (below) that re-runs the logit-lens diversity measurement on your choices. | Where students spend most of their time. |

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
  the 3B thinker (the notebook prints the true layer count). Narrow the window to *localize*
  an effect: block only early layers to test where fusion happens, only late layers to test
  where the answer is composed.

The six token types (modalities) the model tags every position with:

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
| `generated → video` | *If the answer can't look at the frames, does it still describe what's on screen?* | Output changes → the caption was **visually grounded**. Output identical → the model was narrating from audio or priors. |
| `generated → audio` | *If the answer can't hear the soundtrack, does it still describe the sound?* | Output changes → genuine **listening**. Unchanged despite an audio prompt → the "hearing" was cosmetic. |
| `generated → query_text` | *If the answer can't re-read the instruction, does it drift off-task?* | Big drift → the model relies on continuously re-attending to the prompt to stay on task. |
| `audio → video` | *Do the audio tokens borrow from the frames to form their meaning?* (visual → audio fusion) | Diversity at audio positions collapses → the visual stream was actively shaping the audio representations. |
| `video → audio` | *Do the video tokens lean on the soundtrack?* (audio → visual fusion) | Shifts in later behavior → cross-modal binding runs the other way too. |

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
> once the pathway is cut. That is the causal counterpart to the W9 string diff, and where an
> `answer` source becomes meaningful.

### The playground (the tweak-it part)

The final `🎛️` section wraps the diversity measurement in a form — **nothing runs until you
press ▶** — and reuses the already-loaded model, so iterations are fast and need no extra
VRAM. Controls:

- **Video** — upload your own `mp4 / mov / mkv / webm / avi`, or leave it empty to reuse the
  sample clip. (Clips with no audio track produce no audio tokens → nothing to score; the
  notebook says so.)
- **Frames** — 2–32; more frames = richer visual context (and slower).
- **Prompt** — the instruction; try steering it toward sound vs. sight.
- **Knockout on/off**, then either the **single-rule** dropdowns (source, target, layer
  range) or the **advanced** field for several rules as `source,target,start,end` separated
  by `;` (e.g. `audio,video,0,36 ; audio,image,0,36`). The advanced field wins when filled.
- **Compare** — also run a no-knockout baseline so the scoreboard can show a per-layer
  **Δ (knockout − baseline)**.

Reading the scoreboard: each layer is scored by **how many distinct tokens it decodes across
the audio positions**. A **negative Δ** means the knockout made those positions decode
*fewer* distinct tokens — the audio representations got more homogeneous once the blocked
pathway was cut, i.e. that pathway was carrying information. A flat Δ means the intervention
didn't matter (or, for a `generated` source, couldn't).

### Suggested experiments for students

1. **Sight vs. sound, same clip.** Run `generated → video` and then `generated → audio` in
   the knockout cell with the *"see and hear"* prompt. Which knockout changes the caption
   more? What does that say about which sense the model leans on for this clip?
2. **Where does fusion live?** In the playground, run `audio → video` over `[0, 12)`, then
   `[12, 24)`, then `[24, 36)`. Which layer band, when cut, collapses audio diversity the
   most — early, middle, or late?
3. **Prompt steering.** Keep the clip fixed and switch the prompt between *"what you hear"*
   and *"what you see"*. Do the audio positions decode differently even before any knockout?
4. **Bring your own clip.** Upload a video where audio and vision *disagree* (e.g. narration
   over unrelated footage) and see which modality the model reports.
5. **Stack rules.** Use the advanced field to knock out `audio → video` **and**
   `audio → query_text` at once — does starving the audio stream of *both* neighbors compound
   the collapse?

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