# Classroom readiness QA — 2026-09-08

Status: **In progress**. Target: 30 simultaneous participants, up to 1,000 opinions.

Acceptance: roster users can join and submit within one minute; accepted opinions
persist and reach other screens within a few seconds; reconnects do not lose drafts
or strand submission controls; participant traffic excludes reviewer identity;
instructor tools and launch/recovery instructions work.

Tests use synthetic rosters/opinions, temporary databases, and owned local server
processes. No production opinions, credentials, or running class service are changed.

| ID | Scenario / user model | Expected evidence | Result |
| --- | --- | --- | --- |
| B01 | Existing automated regressions | Full suite passes with no unexpected skips | Pending |
| U01 | New participant, wrong ID, normalized ID, phone viewport | Clear errors, successful join, usable form | Pending |
| U02 | Two participants submit Korean, Unicode, markup text | Exact text stored, both maps update, no script execution | Pending |
| U03 | Week filters and theme / viewport changes | Coordinates unchanged and controls reachable | Pending |
| U04 | Network restart while composing | Text, target, week, and source survive | Pending |
| U05 | Connection drops after persistence but before acknowledgement | Form recovers; no duplicate or lost opinion | Pending |
| U06 | Instructor login, search, selection, neighbors, CSV | Correct authorship, results, and download | Pending |
| V01 | WebGPU fallback and viewer reconnect/probe lifecycle | Six opt-in browser regressions pass | Pending |
| V02 | Viewer bundle / disabled backend capability | Working fallback and truthful error states | Pending |
| L01 | 30 simultaneous participants, 1,000-opinion corpus | Latency distribution, exact row counts, responsive heartbeat | Pending |
| L02 | PCA-to-UMAP threshold transition | Correct broadcast and measured delay | Pending |
| P01 | Hostile participant HTTP/SQL/websocket requests | No authorship leakage, mutation, or persistent disruption | Pending |
| P02 | Spreadsheet / HTML-shaped opinions | Safe browser display and instructor export | Pending |
| O01 | Startup, health timeout, roster reload, process recovery | Correct exit codes and no false ready signal | Pending |
| O02 | Intended real embedder and installed dependencies | Offline readiness evidence; limitations explicit | Pending |
| F01 | Build, syntax, lint, artifact/process cleanup | Reproducible commands; unrelated work preserved | Pending |

Final results, defects, fixes, commands, and remaining deployment limits will be
recorded here before the readiness verdict.
