#!/bin/bash
# ============================================================================
# Raaed - Conda Environment Setup Script
# ============================================================================
# This script creates and configures the conda environment with the correct
# installation order to avoid dependency conflicts between PaddleOCR,
# EasyOCR, Sentence-Transformers, ChromaDB, and Docling.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh          # CPU-only (default)
#   ./setup_env.sh gpu      # With CUDA GPU support
# ============================================================================

set -e  # Exit on error

ENV_NAME="raaed"
PYTHON_VERSION="3.10"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_step() { echo -e "\n${BLUE}━━━ $1 ━━━${NC}\n"; }
log_ok()   { echo -e "${GREEN}✓ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
log_err()  { echo -e "${RED}✗ $1${NC}"; }

# ── Check if conda is available ─────────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    log_err "conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# ── Parse arguments ─────────────────────────────────────────────────────────
USE_GPU=false
if [ "$1" = "gpu" ]; then
    USE_GPU=true
    log_warn "GPU mode selected — will install PyTorch with CUDA support"
fi

# ── Remove existing environment if it exists ─────────────────────────────────
log_step "Step 0: Preparing environment"
if conda env list | grep -q "^${ENV_NAME} "; then
    log_warn "Environment '${ENV_NAME}' already exists."
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda deactivate 2>/dev/null || true
        conda env remove -n ${ENV_NAME} -y
        log_ok "Removed existing environment"
    else
        log_warn "Keeping existing environment. Run 'conda activate ${ENV_NAME}' to use it."
        exit 0
    fi
fi

# ── Step 1: Create environment with Python 3.10 ─────────────────────────────
log_step "Step 1: Creating conda environment with Python ${PYTHON_VERSION}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} pip git -y
log_ok "Environment created"

# ── Activate environment ────────────────────────────────────────────────────
log_step "Activating environment"
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
log_ok "Activated ${ENV_NAME} (Python $(python --version 2>&1))"

# ── Step 2: Install PyTorch ─────────────────────────────────────────────────
log_step "Step 2: Installing PyTorch"
if [ "$USE_GPU" = true ]; then
    # Install with CUDA 12.1 support (adjust if needed)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    # CPU-only
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
log_ok "PyTorch installed"

# ── Step 3: Install PaddlePaddle (most restrictive — must go first) ─────────
log_step "Step 3: Installing PaddlePaddle + PaddleOCR"
pip install paddlepaddle>=2.6.0
pip install paddleocr>=2.8.0
log_ok "PaddlePaddle + PaddleOCR installed"

# ── Step 4: Pin critical shared dependencies ─────────────────────────────────
log_step "Step 4: Pinning critical shared dependencies"
pip install "numpy>=1.24.0,<2.0.0" "tokenizers>=0.19.0,<0.30.0"
log_ok "numpy and tokenizers pinned"

# ── Step 5: Install remaining ML/AI packages ─────────────────────────────────
log_step "Step 5: Installing ML/AI packages"
pip install easyocr>=1.7.0
pip install "sentence-transformers>=2.7.0,<4.0.0"
pip install "chromadb>=0.5.0,<1.0.0"
pip install "qdrant-client>=1.9.0,<2.0.0"
pip install tiktoken>=0.7.0
log_ok "ML/AI packages installed"

# ── Step 6: Install PDF extraction tools ─────────────────────────────────────
log_step "Step 6: Installing PDF extraction tools"
pip install pymupdf==1.24.3
pip install "docling>=2.90.0"
pip install "docling-core>=2.0.0"
pip install "pillow>=10.0.0"
log_ok "PDF tools installed"

# ── Step 7: Install web framework & utilities ────────────────────────────────
log_step "Step 7: Installing web framework & utilities"
pip install fastapi==0.110.2
pip install "uvicorn[standard]==0.29.0"
pip install python-multipart==0.0.9
pip install python-dotenv==1.0.1
pip install "pydantic-settings>=2.2.1,<3.0.0"
pip install aiofiles==23.2.1
log_ok "Web framework installed"

# ── Step 8: Install database & LLM providers ─────────────────────────────────
log_step "Step 8: Installing database & LLM providers"
pip install "motor>=3.6.0,<4.0.0"
pip install "pymongo>=4.7.0,<5.0.0"
pip install pydantic-mongo==2.3.0
pip install langchain==0.1.20
pip install langchain-text-splitters
pip install openai==1.35.13
pip install cohere==5.5.8
pip install "ollama>=0.3.0"
log_ok "Database & LLM providers installed"

# ── Step 9: Verify installation ──────────────────────────────────────────────
log_step "Step 9: Verifying installation"

echo "Running dependency check..."
pip check 2>&1 || log_warn "Some dependency warnings detected (may be non-critical)"

echo ""
echo "Testing critical imports..."

python -c "
import sys
results = []

packages = [
    ('fastapi',             'FastAPI'),
    ('uvicorn',             'Uvicorn'),
    ('torch',               'PyTorch'),
    ('sentence_transformers','Sentence-Transformers'),
    ('chromadb',            'ChromaDB'),
    ('pymupdf',             'PyMuPDF (fitz)'),
    ('docling',             'Docling'),
    ('PIL',                 'Pillow'),
    ('openai',              'OpenAI'),
    ('cohere',              'Cohere'),
    ('motor',               'Motor (MongoDB)'),
    ('langchain',           'LangChain'),
    ('tiktoken',            'Tiktoken'),
    ('numpy',               'NumPy'),
]

optional_packages = [
    ('paddleocr',           'PaddleOCR'),
    ('easyocr',             'EasyOCR'),
    ('ollama',              'Ollama'),
    ('qdrant_client',       'Qdrant'),
]

all_ok = True
for module, name in packages:
    try:
        __import__(module)
        version = ''
        try:
            m = __import__(module)
            version = getattr(m, '__version__', '')
        except:
            pass
        print(f'  ✓ {name}: {version}')
    except ImportError as e:
        print(f'  ✗ {name}: FAILED ({e})')
        all_ok = False

print()
print('Optional packages:')
for module, name in optional_packages:
    try:
        __import__(module)
        version = ''
        try:
            m = __import__(module)
            version = getattr(m, '__version__', '')
        except:
            pass
        print(f'  ✓ {name}: {version}')
    except ImportError as e:
        print(f'  ⚠ {name}: not available ({e})')

sys.exit(0 if all_ok else 1)
"

RESULT=$?

echo ""
if [ $RESULT -eq 0 ]; then
    log_ok "All critical packages imported successfully!"
else
    log_err "Some critical packages failed to import. Check the output above."
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  To activate:   conda activate ${ENV_NAME}"
echo "  To run server:  cd src && uvicorn main:app --reload --host 0.0.0.0 --port 5000"
echo "  To deactivate: conda deactivate"
echo ""
