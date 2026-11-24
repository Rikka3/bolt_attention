"""Cell 04 — Stage-3 Chain Distillation (Teacher or Student On-Policy) for Qwen3-4B Bolt-3."""
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
HEALED_CKPT = CKPT_DIR / "bolt3_qwen3_4b_latest.pth"
OUT_PREFIX = "bolt3_qwen3_4b_onpolicy"

CHAIN_SOURCE = "student"   # "student" (on-policy) or "teacher" (sequence-level KD)
ROUNDS = 6
PROMPTS_PER_ROUND = 128
GEN_BS = 16
LR = 5e-5

cmd = [
    sys.executable, "-m", "bolt3", "onpolicy",
    "--ckpt", str(HEALED_CKPT),
    "--donor", DONOR_REPO,
    "--work", str(CKPT_DIR),
    "--out-prefix", OUT_PREFIX,
    "--chain-source", CHAIN_SOURCE,
    "--rounds", str(ROUNDS),
    "--prompts-per-round", str(PROMPTS_PER_ROUND),
    "--gen-bs", str(GEN_BS),
    "--lr", str(LR),
    "--resume",
]
print("[run]", " ".join(cmd))
subprocess.run(cmd, check=True)
