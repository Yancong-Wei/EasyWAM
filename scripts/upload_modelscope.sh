#!/usr/bin/env bash
# Upload a local file or directory to ModelScope.
# Usage: bash scripts/upload_modelscope.sh <local_path> <owner/model> <token>
#
# ModelScope English ids cannot contain spaces. Use WAM_Pretrained_Models.
# Chinese display name defaults to WAM预训练模型.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <local_path> <owner/model> <token>" >&2
  echo "example: $0 ./checkpoints yancow47294/WAM_Pretrained_Models \"\$MODELSCOPE_API_TOKEN\"" >&2
  exit 2
fi

LOCAL="$(readlink -f "$1")"
REPO="$2"
TOKEN="$3"
VISIBILITY="${MODELSCOPE_VISIBILITY:-private}"
CHINESE_NAME="${MODELSCOPE_CHINESE_NAME:-WAM预训练模型}"
DESCRIPTION="${MODELSCOPE_DESCRIPTION:-EasyWAM pretrained world-model checkpoints}"

[[ -e "${LOCAL}" ]] || { echo "path not found: ${LOCAL}" >&2; exit 2; }
[[ "${REPO}" =~ ^[^/]+/[^/]+$ ]] || { echo "repo must be owner/name, got: ${REPO}" >&2; exit 2; }
[[ -n "${TOKEN}" ]] || { echo "token is empty" >&2; exit 2; }

if [[ -z "${OMP_NUM_THREADS:-}" ]] || ! [[ "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  unset OMP_NUM_THREADS || true
fi

MS_BIN="$(command -v ms || true)"
[[ -n "${MS_BIN}" ]] || {
  echo "ms CLI not found. Install: pip install modelscope" >&2
  exit 1
}

export MODELSCOPE_API_TOKEN="${TOKEN}"

echo "login modelscope"
"${MS_BIN}" login --token "${TOKEN}"

echo "upload models | ${LOCAL} -> ${REPO}"
"${MS_BIN}" create "${REPO}" \
  --repo-type model \
  --visibility "${VISIBILITY}" \
  --chinese-name "${CHINESE_NAME}" \
  --description "${DESCRIPTION}" \
  --exist-ok

exec "${MS_BIN}" upload "${REPO}" "${LOCAL}" --repo-type model --use-cache
