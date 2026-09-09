#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077
root="$PWD"
cert=.run/tls/live/feedback-atlas-ip/fullchain.pem
before="$(sha256sum "$cert")"
docker run --rm --user "$(id -u):$(id -g)" --cap-drop ALL \
  --security-opt no-new-privileges \
  --mount "type=bind,src=$root/.run/tls,dst=/etc/letsencrypt" \
  --mount "type=bind,src=$root/.run/tls-work,dst=/var/lib/letsencrypt" \
  --mount "type=bind,src=$root/.run/tls-logs,dst=/var/log/letsencrypt" \
  --mount "type=bind,src=$root/.run/acme-webroot,dst=/var/www/challenges" \
  certbot/certbot:v5.4.0 renew --non-interactive --no-random-sleep-on-renew \
  --cert-name feedback-atlas-ip "$@"
after="$(sha256sum "$cert")"
if [ "$before" != "$after" ]; then
  systemctl --user try-restart feedback-atlas.service
fi
