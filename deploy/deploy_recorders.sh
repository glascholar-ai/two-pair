#!/bin/bash
# Deploy the A1/A2 data recorders to the GCE instance as systemd services.
#
#   deploy/deploy_recorders.sh            # both recorders
#   ONLY=a1 deploy/deploy_recorders.sh    # just one
#
# Recorders are read-only (no API keys, no Secret Manager) and run niced so
# they never compete with the trading services. Remote data/recordings/ is
# never touched by rsync. NOTE: the A1 depth capture needs the disk
# expansion (~50 GB) before it can run for weeks.
set -euo pipefail

REMOTE="${REMOTE:-instance-two-pair}"
DEST="${DEST:-/home/luna/trading}"
ONLY="${ONLY:-}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LOCAL_DIR"

echo "== 1/4 test gate =="
python3 -m pytest tests/test_recorders.py -q

echo "== 2/4 sync code -> $REMOTE:$DEST =="
ssh "$REMOTE" "mkdir -p '$DEST/data/recordings'"
rsync -az --delete \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    recorders "$REMOTE:$DEST/"
rsync -az requirements.txt "$REMOTE:$DEST/"

echo "== 3/4 python venv + deps =="
ssh "$REMOTE" "cd '$DEST' \
    && (test -d venv || python3 -m venv venv) \
    && ./venv/bin/pip install --quiet -r requirements.txt"

install_unit() {
    local name="$1" args="$2"
    ssh "$REMOTE" "sudo tee /etc/systemd/system/$name.service > /dev/null" <<EOF
[Unit]
Description=$name market data recorder
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(ssh "$REMOTE" whoami)
WorkingDirectory=$DEST
ExecStart=$DEST/venv/bin/python $args
Restart=always
RestartSec=10
Nice=10
StandardOutput=append:$DEST/data/$name.log
StandardError=inherit

[Install]
WantedBy=multi-user.target
EOF
    # Manual-start policy: recorders are NOT enabled at boot (user decision
    # 2026-08-17); deploy restarts a running one but never enables it.
    ssh "$REMOTE" "sudo systemctl daemon-reload \
        && sudo systemctl restart $name.service \
        && sleep 2 && systemctl is-active $name.service"
}

echo "== 4/4 systemd units =="
if [[ -z "$ONLY" || "$ONLY" == "a1" ]]; then
    install_unit recorder-a1 \
        "recorders/a1_orderbook.py --config recorders/a1_config.json"
fi
if [[ -z "$ONLY" || "$ONLY" == "a2" ]]; then
    # Binance-only until the IB gateway exists; then add --with-ib here.
    install_unit recorder-a2 \
        "recorders/a2_homeline.py --config recorders/a2_config.json"
fi
echo "done."
