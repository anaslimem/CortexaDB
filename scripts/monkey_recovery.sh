#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${MNEMOS_MONKEY_DIR:-/tmp/mnemos_monkey}"
TOTAL="${MNEMOS_MONKEY_TOTAL:-1000}"
SLEEP_MS="${MNEMOS_MONKEY_SLEEP_MS:-2}"
KILL_AFTER_SEC="${MNEMOS_MONKEY_KILL_AFTER_SEC:-0.35}"

WAL_PATH="${DATA_DIR}/monkey.wal"

cleanup() {
  if [[ -n "${WRITER_PID:-}" ]] && kill -0 "${WRITER_PID}" 2>/dev/null; then
    kill -9 "${WRITER_PID}" || true
  fi
}
trap cleanup EXIT

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"

cd "${ROOT_DIR}"

echo "[0/5] prebuilding binaries"
cargo build --quiet --bin monkey_writer --bin monkey_verify

echo "[1/5] starting writer"
cargo run --quiet --bin monkey_writer -- "${DATA_DIR}" "${TOTAL}" "${SLEEP_MS}" >/tmp/mnemos_monkey_writer.out 2>/tmp/mnemos_monkey_writer.err &
WRITER_PID=$!

sleep "${KILL_AFTER_SEC}"

echo "[2/5] killing writer pid=${WRITER_PID}"
kill -9 "${WRITER_PID}" || true
wait "${WRITER_PID}" 2>/dev/null || true
WRITER_PID=""

for _ in {1..40}; do
  [[ -f "${WAL_PATH}" ]] && break
  sleep 0.05
done

if [[ ! -f "${WAL_PATH}" ]]; then
  echo "WAL not found at ${WAL_PATH}" >&2
  echo "writer stderr:"
  cat /tmp/mnemos_monkey_writer.err 2>/dev/null || true
  exit 1
fi

WAL_SIZE=$(wc -c < "${WAL_PATH}")
if (( WAL_SIZE <= 5 )); then
  echo "WAL too small to truncate safely (size=${WAL_SIZE})" >&2
  exit 1
fi

echo "[3/5] truncating WAL tail by 5 bytes (size ${WAL_SIZE} -> $((WAL_SIZE-5)))"
dd if="${WAL_PATH}" of="${WAL_PATH}.tmp" bs=1 count="$((WAL_SIZE-5))" status=none
mv "${WAL_PATH}.tmp" "${WAL_PATH}"

echo "[4/5] recovering + verifying"
cargo run --quiet --bin monkey_verify -- "${DATA_DIR}"

echo "[5/5] done"
