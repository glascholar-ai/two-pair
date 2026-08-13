#!/bin/bash
# Deploys all three pair services (code sync is idempotent).
#   TG_CHAT_ID=... deploy/deploy_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SERVICE_NAME=twopair         CONFIG=deploy/cfg-skhx.json   deploy/deploy.sh
SERVICE_NAME=twopair-ewysam  CONFIG=deploy/cfg-ewysam.json deploy/deploy.sh
SERVICE_NAME=twopair-mudram  CONFIG=deploy/cfg-mudram.json deploy/deploy.sh
