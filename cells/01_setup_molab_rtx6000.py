"""Cell 01 — Molab RTX 6000 Pro Environment Setup & Bolt-3 Sanity Tests."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
if not (REPO_ROOT / "bolt3").is_dir() and (REPO_ROOT.parent / "bolt3").is_dir():
    REPO_ROOT = REPO_ROOT.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Install missing runtime packages on Molab
REQUIRED_PKGS = [
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("huggingface_hub", "huggingface_hub"),
    ("datasets", "datasets"),
    ("pyarrow", "pyarrow"),
    ("tqdm", "tqdm"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
]
missing = [pip_name for mod, pip_name in REQUIRED_PKGS if importlib.util.find_spec(mod) is None]
if missing:
    print(f"[setup] Installing missing packages: {missing}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

import torch

print(f"[env] Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Repo: {REPO_ROOT}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"[gpu] {props.name} | {props.total_memory / 1024**3:.1f} GB VRAM | bf16={torch.cuda.is_bf16_supported()}")
else:
    print("[gpu] WARNING: CUDA not detected; Bolt-3 training requires a GPU.")

# Optional: set HF_TOKEN if uploading checkpoints or downloading gated models
# os.environ["HF_TOKEN"] = "hf_..."

print("\n[test] Running Bolt-3 parity, gradient, boundary, and checkpoint suite...")
subprocess.run([sys.executable, "-m", "bolt3", "test"], check=True)
