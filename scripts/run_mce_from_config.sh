#!/usr/bin/env bash
# Unified YAML-driven MCE launcher.
#
# Usage:
#   ./scripts/run_mce_from_config.sh configs/runs/my-run.yaml
#
# Dry-run (prints commands without executing):
#   MCE_LAUNCHER_DRY_RUN=1 ./scripts/run_mce_from_config.sh configs/runs/my-run.yaml
#
# The script:
#   1. Sources .env if present (for API keys, model defaults, etc.)
#   2. Delegates to the Python launcher helper which:
#      a. Loads and validates the YAML config
#      b. Runs preflight checks (data files, model)
#      c. Captures a GPU snapshot before execution
#      d. Renders and executes the launch plan (one or two phases)
#   3. Exits with the Python helper's exit code
#
# Environment variables:
#   MCE_LAUNCHER_DRY_RUN=1  Print rendered commands and exit without executing
#
# This script is tmux-friendly: run it inside a tmux session on the remote
# CUDA host and detach safely.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- Source .env for API keys and model defaults ---
if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

# --- Validate arguments ---
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml>" >&2
  echo "" >&2
  echo "Run an MCE training session from a YAML config file." >&2
  echo "" >&2
  echo "Options (via env vars):" >&2
  echo "  MCE_LAUNCHER_DRY_RUN=1  Print commands without executing" >&2
  exit 2
fi

CONFIG_PATH="$1"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Error: config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# --- Delegate to Python launcher ---
exec uv run python -m evo_metaoptics.mce.run_launcher "${CONFIG_PATH}"
