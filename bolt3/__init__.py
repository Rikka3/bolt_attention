"""Bolt-3: Constant-memory Dual-Field Attention for pretrained Transformer LLMs."""
from .attention import Bolt3Attn, Bolt3Decoder, StaticKV
from .converter import convert_donor_model, heal_bolt3_model, onpolicy_distill
from .memory_bank import Bolt3MemoryBank
from .model import FP32_KEYS, Bolt3Block, Bolt3LM, load_bolt3_checkpoint
from .nn import RMSNorm, RoPE, SoftmaxAttn, SwiGLUMLP, repeat_kv

__all__ = [
    "FP32_KEYS",
    "RMSNorm",
    "RoPE",
    "SwiGLUMLP",
    "SoftmaxAttn",
    "repeat_kv",
    "Bolt3Attn",
    "Bolt3Decoder",
    "StaticKV",
    "Bolt3Block",
    "Bolt3LM",
    "load_bolt3_checkpoint",
    "convert_donor_model",
    "heal_bolt3_model",
    "onpolicy_distill",
    "Bolt3MemoryBank",
]
