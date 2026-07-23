# Molab feasibility report

**Overall target-Molab gate:** `UNVERIFIED`
**Local host:** `tx01` (not Molab)

## Evidence boundary

Current official pages advertise Molab public-preview access with 4 CPUs, 32 GB
host RAM, an RTX Pro 6000 Blackwell with 96 GB VRAM, and sessions up to 12 hours:
https://molab.marimo.io/features/vs-colab-alternative and
https://marimo.io/blog/reintroducing-molab. Molab terms describe the service as-is/
as-available and do not provide the resource, persistence, retention, or
availability SLA required for this gate:
https://molab.marimo.io/pages/legal/terms.

The advertised 96 GB GPU is not a 24 GB-class rehearsal. Observing a peak below
24 GB on a 96 GB device cannot reproduce 24 GB allocation, fragmentation, or OOM
behavior. Passing the planned envelope requires a real 24 GB device, an enforceable
hardware-level cap, or an explicitly narrower human-approved claim.

Current subprocessors name CoreWeave for execution and Cloudflare R2 for notebooks
and artifacts (https://marimo.io/pages/legal/subprocessors). An older upload-storage
announcement cannot prove current cache quota, restart survival, deletion, ordinary
retention, or egress controls. `mo.cache` is restart-volatile; persistence under
`mo.persistent_cache` depends on the mounted target filesystem and must be tested:
https://docs.marimo.io/api/caching/.

## Grounded local evidence

- Host RAM: 134,939,664,384 bytes total; 126,354,604,032 bytes available at probe.
- GPUs: two RTX 3090 GPUs, 24,576 MiB each; driver 560.35.03.
- Versions: ffmpeg 4.2.7, PyAV 17.1.0, torch 2.6.0+cu124,
  Transformers 4.52.0, marimo 0.23.14, psutil 7.2.2, qwen-omni-utils 0.0.9.
- Cached model revision: `f75b40e3da2003cdd6e1829b1f420ca70797c34e`.
- Replay/cache regressions: 20 passed; focused committed replay: 1 passed.
- Current bounded process cache uses content SHA + frame count + prompt and FIFO
  `keep=4`; upload identity/cache regressions pass locally.

Local cached-model true-omission forwards passed:

- **Audio omitted:** video is present, `use_audio_in_video=False`, and no audio
  arrays/features/tokens are supplied. Peak allocated VRAM was 9,803,844,608 bytes.
- **Video omitted:** explicit audio-only input with no video pixels/grid/tokens.
  Peak allocated VRAM was 9,582,769,152 bytes.

Those short local probes prove neither the target platform nor the full classroom
route. Omission changes modality/token layouts; position-wise comparisons are
forbidden unless fingerprints are compatible. Otherwise only compatible aggregate
or final-output estimands may be used. Silence is a signal control, not omission.

## PyAV/FFmpeg boundary

PyAV wheels are linked to a particular FFmpeg build; available codecs and muxers
are build-dependent (https://pyav.org/docs/stable/overview/installation.html).
Target encode, flush/mux, reopen/decode, duration/PTS checks, browser playback,
download/re-upload, and processor reingestion have not run. Local H.264/AAC identity
does not establish Molab playback or course rights.

## Mandatory target evidence still absent

1. runtime, GPU, CUDA, driver, CPU/RAM, and image fingerprint;
2. full-route peak host RSS plus CUDA allocated/reserved memory;
3. credible 24 GiB allocation/fragmentation/OOM behavior;
4. cold/warm cache location, size, quota, restart survival, and cleanup;
5. offline replay after population and restart with network disabled;
6. browser upload, temporary copy, deletion, retention, and egress observations;
7. complete PyAV encode/mux/decode/PTS/browser/reingest evidence; and
8. asserted target audio-omitted and video-omitted processor/model traces.

Until every mandatory item passes, local evidence must not be promoted and WP-0
remains incomplete.
