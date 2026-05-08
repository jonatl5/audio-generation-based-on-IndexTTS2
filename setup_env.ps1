$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$IndexTtsRepoDir = if ($env:INDEXTTS_REPO_DIR) {
    $env:INDEXTTS_REPO_DIR
} else {
    Join-Path $ProjectRoot "third_party\index-tts"
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Name"
    }
}

Require-Command git
Require-Command ffmpeg
Require-Command uv
Require-Command python

try {
    git lfs version | Out-Null
} catch {
    throw "Missing git-lfs. Install Git LFS, then rerun this script."
}

Write-Host "Installing this Python project in editable mode..."
Push-Location $ProjectRoot
python -m pip install -e ".[eval,test]"
Pop-Location

Write-Host "Preparing official IndexTTS2 repository at: $IndexTtsRepoDir"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $IndexTtsRepoDir) | Out-Null
if (Test-Path (Join-Path $IndexTtsRepoDir ".git")) {
    Write-Host "Using existing IndexTTS2 repository."
} else {
    git clone https://github.com/index-tts/index-tts.git $IndexTtsRepoDir
}

Push-Location $IndexTtsRepoDir
git lfs install
try {
    git lfs pull
} catch {
    Write-Warning "git lfs pull failed; continuing because checkpoints can still be downloaded separately."
}

Write-Host "Installing IndexTTS2 dependencies with uv sync --all-extras..."
try {
    uv sync --all-extras
} catch {
    Write-Warning "uv sync --all-extras failed. On native Windows this is often caused by optional DeepSpeed dependencies. Falling back to webui extra."
    uv sync --extra webui
}

Write-Host "Installing this dubbing pipeline and evaluation dependencies into the IndexTTS2 uv environment..."
uv pip install -e "$ProjectRoot[eval,test]"

Write-Host "Downloading IndexTTS2 checkpoints..."
try {
    uv tool install "huggingface-hub[cli,hf_xet]"
    hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
    Write-Host "Downloaded checkpoints from Hugging Face."
} catch {
    Write-Warning "Hugging Face download failed; trying ModelScope fallback."
    uv tool install "modelscope"
    modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
}

Write-Host "Running IndexTTS2 GPU check if available..."
try {
    uv run tools/gpu_check.py
} catch {
    Write-Warning "GPU check failed or GPU is unavailable. CPU inference may still work, but it can be slow."
}
Pop-Location

Write-Host ""
Write-Host "Setup complete."
Write-Host ""
Write-Host "Default project config expects:"
Write-Host "  indextts_repo_path: third_party/index-tts"
Write-Host "  indextts_cfg_path: third_party/index-tts/checkpoints/config.yaml"
Write-Host "  indextts_model_dir: third_party/index-tts/checkpoints"
