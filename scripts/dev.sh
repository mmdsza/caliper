#!/usr/bin/env bash
# Start the FastAPI backend and the Next.js editor side-by-side.
# Stops both when either exits or on Ctrl-C.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_PORT="${SERVER_PORT:-8000}"
EDITOR_PORT="${EDITOR_PORT:-3456}"

cleanup() {
  pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(
  cd "$ROOT"
  exec uv run caliper-server --host 127.0.0.1 --port "$SERVER_PORT" --reload
) &
SERVER_PID=$!

(
  cd "$ROOT/packages/editor"
  CALIPER_SERVER_URL="http://127.0.0.1:$SERVER_PORT" \
  PORT="$EDITOR_PORT" \
  exec pnpm dev
) &
EDITOR_PID=$!

echo "[caliper] server pid=$SERVER_PID  http://127.0.0.1:$SERVER_PORT"
echo "[caliper] editor pid=$EDITOR_PID  http://127.0.0.1:$EDITOR_PORT"
echo "[caliper] Ctrl-C to stop both."

# Exit when either child dies. Portable across bash 3.2 (macOS) and bash 4+:
# poll for either PID being gone, with 1s granularity.
while kill -0 "$SERVER_PID" 2>/dev/null && kill -0 "$EDITOR_PID" 2>/dev/null; do
  sleep 1
done

echo "[caliper] one process exited; shutting down the other"
