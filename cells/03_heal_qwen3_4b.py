"""Cell 03 — Heal Qwen3-4B Full Bolt-3 via Knowledge Distillation on Molab RTX 6000 Pro."""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
if not (REPO_ROOT / "bolt3").is_dir() and (REPO_ROOT.parent / "bolt3").is_dir():
    REPO_ROOT = REPO_ROOT.parent
os.chdir(REPO_ROOT)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DONOR_REPO = "Qwen/Qwen3-4B"
CKPT_DIR = REPO_ROOT / "checkpoints"
INIT_CKPT = CKPT_DIR / "bolt3_qwen3_4b_init.pth"
OUT_PREFIX = "bolt3_qwen3_4b"

TOKENS_BUDGET = 16_000_000
STAGE1_TOKENS = 0          # Set e.g. 4_000_000 for layerwise attention warmup before KD
BATCH_SIZE = 8
LR = 8e-5
KD_WEIGHT = 1.0
EVAL_EVERY = 250
SAVE_EVERY = 50
HF_REPO = None             # Optional: e.g. "your-user/bolt3-qwen3-4b" (requires HF_TOKEN)

cmd = [
    sys.executable, "-m", "bolt3", "heal",
    "--ckpt", str(INIT_CKPT),
    "--donor", DONOR_REPO,
    "--work", str(CKPT_DIR),
    "--out-prefix", OUT_PREFIX,
    "--tokens", str(TOKENS_BUDGET),
    "--stage1-tokens", str(STAGE1_TOKENS),
    "--bs", str(BATCH_SIZE),
    "--lr", str(LR),
    "--kd", str(KD_WEIGHT),
    "--eval-every", str(EVAL_EVERY),
    "--save-every", str(SAVE_EVERY),
    "--freeze-mlp",
    "--highspeed",
    "--resume",
]
if HF_REPO:
    cmd.extend(["--hf-repo", HF_REPO])

print("[run]", " ".join(cmd))
subprocess.run(cmd, check=True)
