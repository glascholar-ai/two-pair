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
SERVICE_ARGS="${SERVICE_ARGS:-}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LOCAL_DIR"

echo "== 1/5 test gate =="
python3 -m pytest tests/ -q

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
ssh "$REMOTE" "sudo tee /etc/systemd/system/twopair.service > /dev/null" <<EOF
[Unit]
Description=twopair trading loop
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=luna
WorkingDirectory=$DEST
ExecStart=$DEST/deploy/start.sh $SERVICE_ARGS
Restart=always
RestartSec=30
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF

echo "== 5/5 restart service =="
ssh "$REMOTE" "sudo systemctl daemon-reload \
    && sudo systemctl enable --now twopair \
    && sudo systemctl restart twopair"
sleep 3
ssh "$REMOTE" "systemctl --no-pager --lines=8 status twopair" || true
echo "deploy done. logs: ssh $REMOTE journalctl -u twopair -f"
