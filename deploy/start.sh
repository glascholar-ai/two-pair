#!/bin/bash
# Startup wrapper: fetch Binance credentials from GCP Secret Manager
# (regional secrets, Tokyo) into the process environment, then exec the
# trading loop. Secrets never touch disk or the systemd unit file.
#
# Requires: VM with attached service account granted secretAccessor on the
# secrets below, and access scope "cloud-platform" (see deploy/README.md).
#
# Usage: start.sh [run_live.py args...]     e.g. start.sh --testnet
set -euo pipefail

PROJECT="glascholar"
SECRET_API_KEY="binance-t32-apikey"
SECRET_API_SECRET="binance-t32-secret-key"
# Optional: set to a secret name to enable Telegram notifications.
SECRET_TELEGRAM_TOKEN="${SECRET_TELEGRAM_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

fetch_secret() {
    gcloud secrets versions access latest \
        --secret="$1" --project="$PROJECT"
}

BINANCE_API_KEY="$(fetch_secret "$SECRET_API_KEY")"
BINANCE_API_SECRET="$(fetch_secret "$SECRET_API_SECRET")"
if [[ -z "$BINANCE_API_KEY" || -z "$BINANCE_API_SECRET" ]]; then
    echo "FATAL: fetched empty Binance credentials" >&2
    exit 1
fi
export BINANCE_API_KEY BINANCE_API_SECRET

if [[ -n "$SECRET_TELEGRAM_TOKEN" ]]; then
    TWOPAIR_TELEGRAM_TOKEN="$(fetch_secret "$SECRET_TELEGRAM_TOKEN")"
    export TWOPAIR_TELEGRAM_TOKEN
    export TWOPAIR_TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID"
fi

echo "credentials loaded (key: ${BINANCE_API_KEY:0:4}...); starting loop"
cd "$APP_DIR"
PYTHON="$APP_DIR/venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON="python3"
exec "$PYTHON" run_live.py "$@"
