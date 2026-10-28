# Feedback Atlas Security Deployment

Current classroom URL:

```text
https://143.248.249.7:8888
```

This deployment uses the fixed numeric IP directly. There is no domain and no
Cloudflare tunnel in the normal path.

## Access Model

- One app at `https://143.248.249.7:8888/`, backed by `.run/classroom/atlas.db`
  (`ATLAS_DB` in `.env`). There is no `/demo/` practice app.
- Viewing is open: the map, the Embedding Atlas analysis view and the chat map
  (the home page `/`, read from `.run/chats/atlas.json`; the opinion map is `/map`) need no code. There are no
  personal access codes and no participant sessions.
- Viewing is read-only: the participant websocket accepts no write frame, so
  nothing reaches the corpus from a browser. Opinions arrive only through
  `scripts/seed_ai_opinions.py` plus `SIGUSR1`; chats through
  `scripts/build_chat_atlas.py`.
- Authorship stays private. `reviewer_id` never crosses the participant
  websocket, and the participant `/data/query` database physically lacks the
  reviewer columns. The admin websocket, admin queries and CSV export require the
  admin code in the request body.
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

## Admin Code

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

## Backups

The private data is not in git: `.run/` (classroom database, saved chats and the
chat map, admin code, TLS keys), `roster.csv` and `.env`.
`scripts/backup_data.py` snapshots it into a dated, compressed archive on the
system disk (`~/backups/feedback-atlas/`, a different physical disk from
`/mnt/hdd`), copying each SQLite database through SQLite's backup API and
integrity-checking the copy. The newest 14 archives are kept; they are 0600 in
a 0700 directory.

```bash
.venv/bin/python scripts/backup_data.py                      # one backup now
systemctl --user link "$(pwd)/deploy/feedback-atlas-backup.service"
systemctl --user link "$(pwd)/deploy/feedback-atlas-backup.timer"
systemctl --user daemon-reload
systemctl --user enable --now feedback-atlas-backup.timer   # nightly at 03:30
```

Restore: stop the service, unpack the chosen archive over the project
directory, start the service.

```bash
systemctl --user stop feedback-atlas.service
tar -xzf ~/backups/feedback-atlas/feedback-atlas-<time>.tar.gz -C "$(pwd)"
systemctl --user start feedback-atlas.service
```

Both disks are in this one machine, so this survives a disk failure but not the
loss of the machine. The archives hold student conversations and secrets: copy
them elsewhere only encrypted.

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

- The URL is reachable by anyone who can route to `143.248.249.7:8888`, and
  everything except authorship is readable there without a code, including every
  imported conversation. Import only what students agreed to show this way.
- The admin code is the only credential; rotate it with
  `scripts/configure_security.py` if it is exposed.
- The app is not isolated under a dedicated system user yet.
- Dependency vulnerabilities were not remediated in this documentation-only pass.

## Access Model Change — 2026-10-28

Personal access codes, participant sessions, participant posting, the `/demo/`
practice app and the admin week calendar were removed; the instructor is the only
person adding content. The earlier verification records below describe the
previous model and are kept as history.

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
