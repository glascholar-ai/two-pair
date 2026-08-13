#!/bin/bash
# One-command deploy to the GCE instance.
#
#   deploy/deploy.sh              # test-gate, sync, install deps, (re)start
#   SERVICE_ARGS="--testnet" deploy/deploy.sh   # pass args to run_live.py
#
# Assumes `ssh $REMOTE` works directly and the login user has passwordless
# sudo (GCE default). Never touches remote data/ (journal survives deploys).
set -euo pipefail

REMOTE="${REMOTE:-instance-two-pair}"
DEST="${DEST:-/home/luna/trading}"
SERVICE_NAME="${SERVICE_NAME:-twopair}"
CONFIG="${CONFIG:-deploy/cfg-skhx.json}"
SERVICE_ARGS="${SERVICE_ARGS:---config $CONFIG}"
# Telegram (optional): secret name holding the bot token + the chat id.
TG_SECRET="${TG_SECRET:-telegram-bot-token}"
TG_CHAT_ID="${TG_CHAT_ID:-}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LOCAL_DIR"

echo "== 1/5 test gate =="
python3 -m pytest tests/ -q
python3 - << 'PYGATE'
import glob, json, sys
seen = {}
for f in sorted(glob.glob("deploy/cfg-*.json")):
    cfg = json.load(open(f))
    for key in ("kr_symbol", "us_symbol"):
        s = cfg.get(key, {"kr_symbol": "SKHYNIXUSDT",
                          "us_symbol": "SKHYUSDT"}[key])
        if s in seen:
            sys.exit(f"FATAL: symbol {s} used by both {seen[s]} and {f} — "
                     "two services must never manage the same symbol")
        seen[s] = f
print("pair symbol disjointness OK:", len(seen) // 2, "pairs")
PYGATE

echo "== 2/5 sync code -> $REMOTE:$DEST =="
ssh "$REMOTE" "mkdir -p '$DEST/data'"
rsync -az --delete \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    twopair tests scripts deploy "$REMOTE:$DEST/"
rsync -az run_live.py pair_backtest.py requirements.txt pyrightconfig.json \
    CLAUDE.md README.md "$REMOTE:$DEST/"

echo "== 3/5 python venv + deps =="
ssh "$REMOTE" "cd '$DEST' \
    && (test -d venv || python3 -m venv venv) \
    && ./venv/bin/pip install --quiet --upgrade pip \
    && ./venv/bin/pip install --quiet -r requirements.txt"

echo "== 4/5 systemd unit =="
TG_ENV=""
if [[ -n "$TG_CHAT_ID" ]]; then
    TG_ENV="Environment=SECRET_TELEGRAM_TOKEN=$TG_SECRET
Environment=TELEGRAM_CHAT_ID=$TG_CHAT_ID"
fi
ssh "$REMOTE" "sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null" <<EOF
[Unit]
Description=twopair trading loop ($SERVICE_NAME)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=luna
WorkingDirectory=$DEST
$TG_ENV
ExecStart=$DEST/deploy/start.sh $SERVICE_ARGS
Restart=always
RestartSec=30
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF

echo "== 5/5 restart service =="
ssh "$REMOTE" "sudo systemctl daemon-reload \
    && sudo systemctl enable --now $SERVICE_NAME \
    && sudo systemctl restart $SERVICE_NAME"
sleep 3
ssh "$REMOTE" "systemctl --no-pager --lines=8 status $SERVICE_NAME" || true
echo "deploy done. logs: ssh $REMOTE journalctl -u $SERVICE_NAME -f"
