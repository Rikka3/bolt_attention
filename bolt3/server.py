"""OpenAI-Compatible Local API Server for Bolt-3 (`python -m bolt3 serve`)."""
from __future__ import annotations

import gc
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import torch.nn as nn

from .attention import Bolt3Decoder
from .inspector import fetch_any_donor
from .model import load_bolt3_checkpoint


def get_all_eos_ids(tokenizer) -> set[int]:
    eos_ids = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.update(int(x) for x in tokenizer.eos_token_id)
        else:
            eos_ids.add(int(tokenizer.eos_token_id))
    for tok_str in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<｜end▁of▁sentence｜>", "</s>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1):
                eos_ids.add(tid)
        except Exception:
            pass
    return eos_ids


def _quantize_cpu_low_ram(model):
    q_fn = getattr(getattr(torch, "ao", torch), "quantization", torch.quantization).quantize_dynamic
    model.embed_tokens.weight.data = model.embed_tokens.weight.data.to(torch.bfloat16)
    model.embed_tokens.register_forward_hook(lambda _m, _in, out: out.float())
    model.norm.float()

    head_w = model.lm_head.weight.detach().float()
    tmp_head = nn.Linear(model.lm_head.in_features, model.lm_head.out_features, bias=False)
    tmp_head.weight = nn.Parameter(head_w)
    model.lm_head = q_fn(nn.ModuleDict({"h": tmp_head}), {nn.Linear}, dtype=torch.qint8)["h"]
    del tmp_head, head_w
    gc.collect()

    for li in range(len(model.blocks)):
        blk = model.blocks[li].float()
        model.blocks[li] = q_fn(blk, {nn.Linear}, dtype=torch.qint8)
        del blk
        if li % 4 == 3:
            gc.collect()
    gc.collect()
    return model.eval()


def load_model_for_serving(ckpt_path: str, donor_tokenizer: str | None = None, quantize_cpu: bool = True):
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bolt3-serve] Loading {ckpt_path} on {device}...")
    model, _, ck_donor, _, _ = load_bolt3_checkpoint(ckpt_path, device=device, donor_override=donor_tokenizer, eval_mode=True)
    donor_id = donor_tokenizer or ck_donor
    if not donor_id:
        raise ValueError("Checkpoint lacks donor metadata; pass --donor <hf_repo_id> explicitly.")

    if device.type == "cpu":
        if quantize_cpu:
            print("[bolt3-serve] Applying low-RAM layerwise INT8 CPU quantization...")
            model = _quantize_cpu_low_ram(model)
        else:
            model = model.float().eval()

    donor_dir = fetch_any_donor(donor_id)
    tokenizer = AutoTokenizer.from_pretrained(donor_dir)
    print(f"[bolt3-serve] Bolt-3 & tokenizer ({donor_id}) ready on {device}!")
    return model, tokenizer, device


@torch.no_grad()
def generate_completion(
    model,
    tokenizer,
    device,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
    decoder: Bolt3Decoder | None = None,
) -> str:
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant: "

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(ids)
    eos_ids = get_all_eos_ids(tokenizer)
    if decoder is None:
        decoder = Bolt3Decoder(model)

    prompt_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    out_tensor = decoder.generate(
        prompt_tensor,
        max_new_tokens=max_tokens,
        temperature=temperature,
        eos_ids=eos_ids,
        repetition_penalty=repetition_penalty,
    )
    gen_ids = out_tensor[0, prompt_len:].tolist()
    if eos_ids and gen_ids and gen_ids[-1] in eos_ids:
        gen_ids = gen_ids[:-1]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def run_server(
    ckpt_path: str,
    donor_tokenizer: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    quantize_cpu: bool = False,
):
    model, tokenizer, device = load_model_for_serving(ckpt_path, donor_tokenizer, quantize_cpu=quantize_cpu)
    decoder = Bolt3Decoder(model)

    class OpenAIHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/v1/models"):
                payload = {"object": "list", "data": [{"id": "bolt3", "object": "model", "owned_by": "bolt3"}]}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Bolt-3 OpenAI-Compatible Server Running")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            messages = body.get("messages", [{"role": "user", "content": body.get("prompt", "")}])
            max_tokens = int(body.get("max_tokens", 1024))
            temperature = float(body.get("temperature", 0.0))
            rep_pen = float(body.get("repetition_penalty", 1.0))

            reply = generate_completion(
                model,
                tokenizer,
                device,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=rep_pen,
                decoder=decoder,
            )
            resp = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "bolt3"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
            }
            data = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print("=" * 78)
    print(f"Bolt-3 OpenAI API Server listening at http://{host}:{port}/v1")
    print("=" * 78)
    HTTPServer((host, port), OpenAIHandler).serve_forever()
