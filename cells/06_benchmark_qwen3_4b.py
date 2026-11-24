"""Cell 06 — Full 7-Task Benchmark + 2k-64k Throughput & Constant-RAM Scaling vs Qwen/Qwen3-4B."""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
if not (REPO_ROOT / "bolt3").is_dir() and (REPO_ROOT.parent / "bolt3").is_dir():
    REPO_ROOT = REPO_ROOT.parent
os.chdir(REPO_ROOT)

DONOR_REPO = "Qwen/Qwen3-4B"
CKPT_DIR = REPO_ROOT / "checkpoints"
ONPOLICY_CKPT = CKPT_DIR / "bolt3_qwen3_4b_onpolicy_latest.pth"
HEALED_CKPT = CKPT_DIR / "bolt3_qwen3_4b_latest.pth"
CKPT = ONPOLICY_CKPT if ONPOLICY_CKPT.is_file() else HEALED_CKPT

LIMIT = 200
GSM8K_LIMIT = 200
COMPARE_DONOR = True

cmd = [
    sys.executable, "-m", "bolt3", "benchmark",
    "--ckpt", str(CKPT),
    "--donor", DONOR_REPO,
    "--limit", str(LIMIT),
    "--gsm8k-limit", str(GSM8K_LIMIT),
]
if not COMPARE_DONOR:
    cmd.append("--no-donor-compare")

print("[run]", " ".join(cmd))
subprocess.run(cmd, check=True)
