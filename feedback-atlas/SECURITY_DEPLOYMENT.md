# Feedback Atlas Security Deployment

Current classroom URL:

```text
https://143.248.249.7:8888
```

This deployment uses the fixed numeric IP directly. There is no domain and no
Cloudflare tunnel in the normal path.

## Classroom And Demo

- Classroom: `https://143.248.249.7:8888/`, for weeks 1–4, using private codes
  and `.run/classroom/atlas.db` (`ATLAS_DB` in `.env`).
- Practice: `https://143.248.249.7:8888/demo/`, using one shared `demo` account
  and a single entry button, with no personal-code input. Historical sample
  names remain in `demo/roster.csv` for targets and existing feedback attribution;
  `.run/demo/atlas.db` (`ATLAS_DEMO_DB`) stores the practice corpus.
- The original 30 example opinions belong in the demo. The classroom starts
  empty. The previous `atlas.db` is retained privately as a pre-separation backup;
  it is no longer the active classroom database.
- The demo roster and credentials are independent of the classroom roster and
  private codes. Session tokens, WebSockets, queries, and exports are scoped to
  their respective app; neither app accepts the other's credentials.
- Demo comments are shared practice content, capped at 500 opinions. The demo
  admin code displayed in the demo unlocks only practice data.
- Main and demo share one serialized embedding model but have separate corpus,
  layout, database, and connection state.

The one-time migration must run while the service is stopped:

```bash
systemctl --user stop feedback-atlas.service
.venv/bin/python scripts/separate_demo_data.py --source atlas.db \
  --demo .run/demo/atlas.db --classroom .run/classroom/atlas.db \
  --expected-opinions 30
systemctl --user start feedback-atlas.service
```

The script verifies copies and counts before changing `.env`. It refuses an
unexpected source count or existing destination files. It does not regenerate
participant codes or edit the classroom roster. Future real-class roster changes
belong in `roster.csv`; demo names remain in `demo/roster.csv`.

## Security State

The main classroom uses server-time week assignment (`ATLAS_AUTO_WEEK=1` by
default). The admin calendar's four inclusive seven-day ranges are:

| Week | Start (Asia/Seoul) | Last included day |
| --- | --- | --- |
| 1 | 2026-10-15 | 2026-10-21 |
| 2 | 2026-10-22 | 2026-10-28 |
| 3 | 2026-11-05 | 2026-11-11 |
| 4 | 2026-11-12 | 2026-11-18 |

The midterm gap is October 29–November 4. A new submission outside all ranges is
refused; no out-of-period record is silently assigned to a week. Demo practice
remains available as week 1. Client week/timestamp overrides cannot change the
main server's classification. Accepted receipt retries remain idempotent even
after the window closes.

`GET /api/schedule` exposes only calendar metadata. Admin-only `PUT /api/schedule`
uses the existing access code plus an expected revision; settings are saved in
each app's own SQLite `meta.class_schedule_v1` value. Calendar edits apply only
to future submissions. Conflicting admin changes return 409 without overwriting
the newer settings. `ATLAS_AUTO_WEEK=0` is a legacy/test opt-out, not the active
classroom deployment setting.

Schedule rollout verification: 333 backend regression tests and 28 browser
checks passed, including KST boundaries, the midterm gap, client timestamp/week
override rejection, idempotent receipts after closing, admin authorization,
revision conflicts, and real calendar-button saves. The live calendar was saved
at revision 1 with the specified dates and verified in SQLite. Live mobile and
dark-mode controls and both compose placeholders were checked. Main remains at
zero opinions and demo at 30; no live test feedback was added.

- Participant entry requires only that person's fixed private access code. The
  server resolves its enrolled owner and role; there is no ID input field. Codes
  remain unchanged across server restarts. Duplicate codes are rejected.
- `.run/access-codes.json` stores SHA-256 hashes only. The cleartext
  `.run/access-codes.csv` is for private distribution and must not be posted to a
  shared channel.
- `POST /data/query` requires `Authorization: Bearer <session_token>` for
  participant data. Admin data still requires the admin code in the request body.
- Plain HTTP is rejected by default. `ATLAS_ALLOW_INSECURE_HTTP=1` is for local
  development and tests only.
- The trusted Let's Encrypt IP certificate is stored under
  `.run/tls/live/feedback-atlas-ip/`.
- No classroom IP-range firewall restriction is installed because no known
  classroom CIDR is available.
- Dedicated OS-user isolation is not configured; the service currently runs under
  the existing `june` account.
- Application dependencies were not changed by this deployment, and the
  dependency advisory scan has not been rerun in this documentation pass.

## Prepare Credentials

Generate per-person codes after `roster.csv` is ready:

```bash
python scripts/create_access_codes.py \
  --roster roster.csv \
  --hashes .run/access-codes.json \
  --out .run/access-codes.csv
```

Give each participant only their own row from `.run/access-codes.csv`. Do not
share the whole CSV publicly.

If the old admin code may have crossed plain HTTP, rotate it once:

```bash
python scripts/configure_security.py
```

The rotated admin code is stored in `.run/admin-code.txt`. Keep that file
private; do not copy its contents into docs, slides, or chat.

## Start And Stop

Install the user services from the repository:

```bash
systemctl --user link "$(pwd)/deploy/feedback-atlas.service"
systemctl --user link "$(pwd)/deploy/feedback-atlas-renew.service"
systemctl --user link "$(pwd)/deploy/feedback-atlas-renew.timer"
systemctl --user daemon-reload
systemctl --user enable --now feedback-atlas.service feedback-atlas-renew.timer
loginctl enable-linger "$USER"
```

The live host has these units linked and enabled, with linger enabled for the
user service.

Use the classroom launcher during class:

```bash
scripts/class.sh start
scripts/class.sh url
scripts/class.sh status
scripts/class.sh stop
```

`scripts/class.sh` starts, checks, and stops the managed HTTPS service. It does
not start a Cloudflare tunnel.

Check status and logs directly when needed:

```bash
systemctl --user status feedback-atlas.service
systemctl --user status feedback-atlas-renew.timer
journalctl --user -u feedback-atlas.service -f
```

Stop or restart:

```bash
systemctl --user stop feedback-atlas.service
systemctl --user restart feedback-atlas.service
```

Local health check:

```bash
curl --noproxy '*' --resolve 143.248.249.7:8888:127.0.0.1 https://143.248.249.7:8888/healthz
```

This routes the request to loopback while preserving certificate validation for
`143.248.249.7`.

## Certificate Renewal

The renewal timer runs twice daily:

```bash
systemctl --user list-timers feedback-atlas-renew.timer
systemctl --user start feedback-atlas-renew.service
journalctl --user -u feedback-atlas-renew.service -n 80
```

`scripts/renew_ip_certificate.sh` restarts `feedback-atlas.service` only when the
certificate file changes.

The helper for HTTP-01 challenges is isolated in
`deploy/ip-https.compose.yml`. It runs `certbot/certbot:v5.4.0`, mounts only the
ACME challenge webroot, and uses the existing Traefik 2.9.10 IP-only rule:

```text
Host(`143.248.249.7`) && PathPrefix(`/.well-known/acme-challenge/`)
```

That router uses priority `2147483648` so the IP challenge path wins without
changing shared domain routes. Reverify the challenge route after any Traefik or
proxy upgrade because router priority behavior is proxy-version sensitive.

## Remaining Risks

- The URL is reachable by anyone who can route to `143.248.249.7:8888`; endpoint
  use is protected by participant codes, signed sessions, and the admin code.
- A participant code can be shared or stolen. Rotate access codes if one is
  exposed.
- The app is not isolated under a dedicated system user yet.
- Dependency vulnerabilities were not remediated in this documentation-only pass.

## Initial Security Deployment Verification — 2026-09-09

- Full repository suite: 284 passed; 19 browser checks passed, including three
  real-browser checks with CSP enabled. JavaScript and shell syntax checks,
  Python compilation, systemd unit validation, and `git diff --check` passed.
- Live HTTPS certificate verification succeeded without bypassing trust checks.
  All three TA accounts authenticated with their private codes and received the
  participant snapshot; authenticated viewer queries returned the preserved 30
  rows. ID-only entry and unauthenticated queries were denied.
- The rotated admin credential was verified without printing it. Credentials,
  roster, and database files have owner-only permissions.
- Let's Encrypt staging renewal succeeded. The installed renewal service also
  exited successfully, its timer is active, and user lingering is enabled.
- The existing website continued to return its original HTTP 302 response.
- A private pre-deployment database backup is in `.run/atlas-before-security.db`.

Implementation files: `src/access.py`, `src/transport_security.py`,
`src/server.py`, `src/config.py`, `src/hub.py`, `web/index.html`, `web/app.js`,
`web/viewer.js`, credential/setup scripts under `scripts/`, the HTTPS classroom
launcher and renewal script, and service definitions under `deploy/`. Related
tests and both README translations were updated. The old direct public HTTP
entrypoint was disabled, and the classroom launcher now uses one managed HTTPS
service instead of managing separate server and tunnel processes.

## Demo Separation Verification — 2026-09-09

- Main now contains zero opinions; demo contains all 30 original example opinions.
  Every original opinion field was compared with the retained source database.
- The live demo exposes 13 public practice accounts; main credentials were not
  regenerated. Demo and main session tokens and admin codes were rejected when
  used against the opposite app. Authenticated queries returned 30 and 0 rows,
  respectively, and unauthenticated queries were denied.
- Full regression run: 309 passed. The migration tests were then extended with
  missing-config and failed-config-swap recovery cases; all six migration tests
  passed. The combined browser collection passed 23 cases.
- A live mobile browser verified the demo redirect, 13-account selector, 30 map
  points, persistent demo banner, and separate main sign-in. No test feedback was
  added to the live datasets.
- Shared frontend assets now use scoped URLs/storage in `web/app-context.js`.
  `src/demo.py`, `demo/roster.csv`, the explicit mounted-app lifecycle, and
  `scripts/separate_demo_data.py` implement the separation without duplicating
  the classroom UI. Both apps use the same serialized embedding model.
- A manual tunnel is outside this runbook. If one is used later, keep the same
  credentials, HTTPS browser origin, websocket upgrade, and origin checks.
