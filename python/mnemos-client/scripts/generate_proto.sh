#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="${ROOT_DIR}/../../proto"
OUT_DIR="${ROOT_DIR}/src/mnemos_client/proto"

python3 -m grpc_tools.protoc \
  -I"${PROTO_DIR}" \
  --python_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  "${PROTO_DIR}/mnemos.proto"

# Ensure package-relative imports in generated grpc module.
OUT_FILE="${OUT_DIR}/mnemos_pb2_grpc.py"
if [[ -f "${OUT_FILE}" ]]; then
  python3 - << PY
from pathlib import Path
p = Path(r"${OUT_FILE}")
s = p.read_text()
s = s.replace("import mnemos_pb2 as mnemos__pb2", "from . import mnemos_pb2 as mnemos__pb2")
p.write_text(s)
PY
fi

echo "Generated gRPC stubs in ${OUT_DIR}"
