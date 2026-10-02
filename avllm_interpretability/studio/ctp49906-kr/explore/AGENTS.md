# CTP49906 Korean Studio View

This directory contains the Vanilla marimo-studio view named `explore` for the Korean AVLLM lab.

Authoring rules:

- Keep all projection hosts inside the single `#app-shell` element in `index.html`.
- Use native Studio projection elements for notebook state and controls: `marimo-cell`, `marimo-output`, and `mo-value`.
- Do not add JavaScript RPC calls, notebook setters, custom form reconstruction, or duplicated projection hosts.
- Mount each complete-cell target at most once in this presentation. Change visibility with CSS/JavaScript only.
- Keep CSS and JavaScript local to this view. Do not add external fonts, CDNs, or framework dependencies.
- Do not create `.owners` records manually; let Studio discover and manage ownership.
- Generated `.artifacts/` and `.locks/` directories are ignored by the parent `.gitignore`.

The notebook provides these canonical complete-cell targets:

`studio_status`, `diversity_form`, `diversity_result_panel`, `band_form`, `band_result_panel`, `tf_form`, `tf_result_panel`, `tf_threshold_panel`, `tf_tokens_panel`, `lab_intro`, `method_guide`, `clip_preview`, `token_census`, `probe_grid_panel`, `probe_summary_panel`, `guided_captions`, `attention_mass_panel`, `guided_tf_panel`, `guided_tf_threshold`, `guided_tf_tokens`, `ledger_panel`, `verdict_panel`, `verdict_submit_handler`, and `worksheet_panel`.
