#!/usr/bin/env bash
set -euo pipefail

# Colours for terminal output
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
RESET='\033[0m'

if [[ ! -t 1 ]]; then
    BOLD='' GREEN='' YELLOW='' CYAN='' RED='' RESET=''
fi

# Short helpers so that the rest of the script isnt drowning in escape codes
print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════${RESET}"
    echo -e "${CYAN}${BOLD}  $1${RESET}"
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════${RESET}"
}

print_ok()   { echo -e "  ${GREEN}[OK]${RESET}    $1"; }
print_warn() { echo -e "  ${YELLOW}[!]${RESET}     $1"; }
print_err()  { echo -e "  ${RED}[ERROR]${RESET} $1"; }
print_info() { echo -e "          $1"; }

# Asks the user a y/n question. Defaults to yes if they just hit enter.
ask_confirm() {
    local prompt="$1"
    local reply
    echo -en "\n  ${BOLD}--> ${prompt} [Y/n]${RESET} "
    read -r reply
    reply="${reply:-Y}"
    [[ "$reply" =~ ^[Yy]$ ]]
}

# Parses the --ci flag. CI runners (e.g. Github runners) don't have stdin, so this auto-approves everyhing
CI_MODE=false
for arg in "$@"; do
    case "$arg" in
        --ci) CI_MODE=true ;;
    esac
done

if $CI_MODE; then
    ask_confirm() { return 0; }
fi

# Figures out where this script lives, so all the relative paths
# (pyproject.toml, .venv, etc.) work even if caleld from somewhere else
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


# ---- Welcome ----------------------------------------------------------------

print_header "Harmful Content Detection - Project Setup"
echo ""
echo -e "  ${BOLD}Author:${RESET}      Samuel Ruairí Bullard"
echo -e "  ${BOLD}Project:${RESET}     Evaluating Cost-Sensitive Loss Functions for"
echo -e "               Transformer-Based German Harmful Content Detection"
echo -e "  ${BOLD}Institution:${RESET} University of Regensburg"
echo ""
echo -e "  This script sets up the Python environment for the project codebase."
echo -e "  It uses ${BOLD}uv${RESET} (https://docs.astral.sh/uv/) for fast and reproducible"
echo -e "  dependency management."
echo ""
echo -e "  ${BOLD}What this script will do:${RESET}"
echo -e "    1. Install the uv package manager (if not present)"
echo -e "    2. Set up Python 3.10 + all dependencies via ${BOLD}uv sync${RESET}"
echo -e "    3. Install the convokit package"
echo -e "    4. Detect GPU acceleration capabilities"
echo -e "    5. Verify all imports"
echo -e "    6. Print the next steps"
echo ""

ask_confirm "Proceed with setup?" || { echo "  Setup cancelled."; exit 0; }


# ---- Step 1: uv -------------------------------------------------------------

print_header "Step 1 - Package Manager (uv)"

if command -v uv &>/dev/null; then
    print_ok "uv found: $(uv --version)"
else
    print_warn "uv is not installed."
    print_info "uv is a fast Python package manager by Astral."
    print_info "More info: https://docs.astral.sh/uv/"
    echo ""

    ask_confirm "Install uv now?" || { print_err "uv is required. Aborting."; exit 1; }

    curl -LsSf https://astral.sh/uv/install.sh | sh

    # uvs installer drops the binary into the ~/.local/bin path, so we need to make
    # sure the current shell can find it for the rest of this script via
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &>/dev/null; then
        print_ok "uv installed: $(uv --version)"
    else
        print_err "uv installation failed. Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi


# ---- Step 2: Python + deps --------------------------------------------------
# uv reads the .python-version (pinned to 3.10) and pyproject.toml, then takes care
# of downloading the right Python build, creating the venv, and installing
# everything from the lockfile.

print_header "Step 2 - Python 3.10 + Dependencies"

echo ""
echo -e "  Running ${BOLD}uv sync${RESET} in ${SCRIPT_DIR}"
echo -e "  This will:"
echo -e "    - Download Python 3.10 (if not already installed)"
echo -e "    - Create a virtual environment (.venv)"
echo -e "    - Install all Python packages from pyproject.toml"
echo ""
echo -e "  Largest download: PyTorch (~2 GB on first install)."

ask_confirm "Proceed with installation?" || { print_err "Aborted."; exit 1; }

echo ""
uv sync --directory "$SCRIPT_DIR"
print_ok "All dependencies installed."

# convokit 4.0.0 has a huge dependency tree (~5 GB) that pulls in tensorflow,
# unsloth and a bunch of other stuff we never actually use in the project's codebase.
# To avoid all of that bloat we install it with --no-deps and just rely on the
# 3 runtime deps it actually needs (pymongo, requests, tqdm), which are already
# listed in the pyproject.toml.
echo ""
print_info "Installing convokit==4.0.0 (--no-deps)..."
uv pip install --no-deps convokit==4.0.0 --directory "$SCRIPT_DIR"
print_ok "convokit installed (minimal dependencies)."


# ---- Step 3: GPU info -------------------------------------------------------
# Tells the user what acceleration is available on their machine.

print_header "Step 3 - GPU Acceleration"

OS_NAME="$(uname -s)"
ARCH="$(uname -m)"

case "$OS_NAME" in
    Linux*)
        if command -v nvcc &>/dev/null; then
            CUDA_VER="$(nvcc --version | grep 'release' | sed 's/.*release //' | cut -d',' -f1)"
            print_ok "CUDA found (version ${CUDA_VER})."
        else
            print_warn "CUDA (nvcc) not found in PATH."
            echo ""
            echo -e "  GPU training will not be available without CUDA."
            echo -e "  PyTorch will run on CPU only."
            echo ""
            echo -e "  To enable GPU acceleration:"
            echo -e "    1. Install the NVIDIA driver for your GPU"
            echo -e "    2. Install the CUDA Toolkit: ${BOLD}https://developer.nvidia.com/cuda-downloads${RESET}"
            echo -e "    3. Verify with: nvcc --version"
        fi
        ;;
    Darwin*)
        if [[ "$ARCH" == "arm64" ]]; then
            print_ok "Apple Silicon detected, MPS (Metal) GPU acceleration available."
            print_info "MPS is suitable for inference and evaluation."
            print_info "Full training was performed on Linux with NVIDIA CUDA."
        else
            print_warn "Intel Mac, CPU only. No GPU acceleration available."
        fi
        ;;
esac


# ---- Step 4: Verification ---------------------------------------------------
# Tries to import every package we need.
# Installation Success Verification essentially.

print_header "Step 4 - Verifying Installation"

VERIFY_RESULT=$(uv run --directory "$SCRIPT_DIR" python -c '
import sys

results = []

def check(name, code):
    try:
        exec(code)
        results.append(("OK", name))
    except Exception as e:
        results.append(("FAIL", f"{name}: {e}"))

check("torch",          "import torch")
check("transformers",   "from transformers import AutoTokenizer")
check("datasets",       "from datasets import load_dataset")
check("scikit-learn",   "from sklearn.metrics import f1_score")
check("pandas",         "import pandas")
check("numpy",          "import numpy")
check("wandb",          "import wandb")
check("scipy",          "from scipy.stats import bootstrap")
check("statsmodels",    "import statsmodels")
check("torchmetrics",   "from torchmetrics.classification import MulticlassCalibrationError")
check("convokit",       "from convokit import Corpus, Speaker, Utterance, FightingWords")
check("matplotlib",     "import matplotlib")
check("seaborn",        "import seaborn")
check("rich",           "from rich.console import Console")
check("jupyter",        "import jupyter_core")
check("nltk",           "import nltk")
check("pyyaml",         "import yaml")
check("python-dotenv",  "from dotenv import load_dotenv")
check("evaluate",       "import evaluate")

ok_count  = sum(1 for s, _ in results if s == "OK")
fail_count = sum(1 for s, _ in results if s == "FAIL")
total = len(results)

print(f"\n  Verification: {ok_count}/{total} packages imported successfully\n")
for status, msg in results:
    if status == "OK":
        print(f"  \033[0;32m[OK]\033[0m    {msg}")
    else:
        print(f"  \033[0;31m[FAIL]\033[0m  {msg}")

if fail_count > 0:
    print(f"\n  \033[1;33m{fail_count} package(s) failed. Review the errors above.\033[0m")
    sys.exit(1)
else:
    print(f"\n  \033[0;32mAll checks passed.\033[0m")
') || true

echo "$VERIFY_RESULT"

if uv run --directory "$SCRIPT_DIR" python -c '
import sys
try:
    import torch, transformers, datasets, sklearn, pandas, numpy
    import wandb, scipy, statsmodels, torchmetrics, matplotlib, seaborn
    import rich, nltk, yaml, evaluate
    from dotenv import load_dotenv
    from convokit import Corpus, Speaker, Utterance, FightingWords
except ImportError as e:
    sys.exit(1)
' 2>/dev/null; then
    VERIFY_PASSED=true
else
    VERIFY_PASSED=false
fi


# ---- Done --------------------------------------------------------------------

print_header "Setup Complete"

if $VERIFY_PASSED; then
    echo -e "\n  ${GREEN}${BOLD}All packages installed and verified successfully.${RESET}"
else
    echo -e "\n  ${YELLOW}${BOLD}Setup finished with verification warnings. See above.${RESET}"
fi

echo ""
echo -e "  ${BOLD}Next steps:${RESET}"
echo ""
echo -e "  1. ${BOLD}Activate the virtual environment${RESET}:"
echo -e "       source .venv/bin/activate"
echo -e "       ${BOLD}or${RESET} prefix commands with: uv run <command>"
echo ""
echo -e "  2. ${BOLD}Configure Weights & Biases${RESET} (optional):"
echo -e "       Create a .env file:  echo 'WANDB_API_KEY=\"your-key\"' > .env"
echo -e "       Or disable W&B:      export WANDB_MODE=offline"
echo ""
echo -e "  3. ${BOLD}Run notebooks in order${RESET}:"
echo -e "       notebooks/train/01_data_preprocessing/00_data_preprocessing.ipynb"
echo -e "       notebooks/train/02_model_training/"
echo -e "       notebooks/train/03_model_evaluation/"
echo ""
echo -e "  4. ${BOLD}Finetuned model weights${RESET} (9 models, ~515 MB each):"
echo -e "       models/finetuned_models/{c2a,dbo,vio}/ contain the checkpoints"
echo -e "       for all 3 subtasks x 3 loss strategies (Baseline, CWCE, CW+FL)."
echo -e ""
echo -e "       ${BOLD}If the project is from the Datentraeger (USB):${RESET}"
echo -e "         Model weights are in the sibling folder ${BOLD}3_Modelle/${RESET}."
echo -e "         Finetuned models: 3_Modelle/{c2a,dbo,vio}/"
echo -e "         Base models:      3_Modelle/base_models/{EuroBERT-210m,ModernGBERT_134M}/"
echo -e ""
echo -e "       ${BOLD}If the project was cloned from GitHub:${RESET}"
echo -e "         Model weights are stored with Git LFS. If the files in"
echo -e "         models/finetuned_models/ are small pointer files (~130 bytes)"
echo -e "         instead of ~515 MB .safetensors, run:"
echo -e "           git lfs pull"
echo ""

case "$OS_NAME" in
    Darwin*)
        echo -e "  ${YELLOW}${BOLD}macOS note:${RESET} Training was performed on Linux with NVIDIA CUDA."
        echo -e "  Training notebooks will run on MPS/CPU but probably much slower. For evaluation-only"
        echo -e "  use, the existing finetuned model checkpoints are sufficient and will run fast enough"
        echo -e "  even on a CPU."
        echo ""
        ;;
esac

echo -e "  Deactivate the virtual environment when done: ${BOLD}deactivate${RESET}"
echo ""
