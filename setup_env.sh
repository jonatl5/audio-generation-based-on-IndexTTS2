#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEXTTS_REPO_DIR="${INDEXTTS_REPO_DIR:-${PROJECT_ROOT}/third_party/index-tts}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_cmd git
need_cmd ffmpeg
need_cmd uv
need_cmd python

if ! git lfs version >/dev/null 2>&1; then
  echo "Missing git-lfs. Install it, then rerun this script." >&2
  exit 1
fi

echo "Installing this Python project in editable mode..."
cd "${PROJECT_ROOT}"
python -m pip install -e ".[eval,test]"

echo "Preparing official IndexTTS2 repository at: ${INDEXTTS_REPO_DIR}"
mkdir -p "$(dirname "${INDEXTTS_REPO_DIR}")"
if [ -d "${INDEXTTS_REPO_DIR}/.git" ]; then
  echo "Using existing IndexTTS2 repository."
else
  git clone https://github.com/index-tts/index-tts.git "${INDEXTTS_REPO_DIR}"
fi

cd "${INDEXTTS_REPO_DIR}"
git lfs install
git lfs pull || true

echo "Installing IndexTTS2 dependencies with uv sync --all-extras..."
if ! uv sync --all-extras; then
  cat <<'EOF'

uv sync --all-extras failed. On native Windows this is commonly caused by the
optional DeepSpeed extra requiring CUDA_HOME and a local CUDA build toolchain.
Continuing with the non-DeepSpeed environment.

EOF
  uv sync --extra webui
fi

echo "Installing this dubbing pipeline and evaluation dependencies into the IndexTTS2 uv environment..."
uv pip install -e "${PROJECT_ROOT}[eval,test]"

echo "Downloading IndexTTS2 checkpoints..."
if uv tool install "huggingface-hub[cli,hf_xet]" && hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints; then
  echo "Downloaded checkpoints from Hugging Face."
else
  echo "Hugging Face download failed; trying ModelScope fallback..."
  uv tool install "modelscope"
  modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
fi

echo "Running IndexTTS2 GPU check if available..."
uv run tools/gpu_check.py || true

cat <<EOF

Setup complete.

Default project config expects:
  indextts_repo_path: third_party/index-tts
  indextts_cfg_path: third_party/index-tts/checkpoints/config.yaml
  indextts_model_dir: third_party/index-tts/checkpoints

For Windows users: run this from Git Bash, WSL, or another Bash-compatible shell.
DeepSpeed can be difficult on native Windows; rerun IndexTTS2's uv sync without
--all-extras if DeepSpeed dependencies block installation.
EOF
