"""Decode-parity and KL drift-by-position diagnostics (`python -m bolt3 diagnose`)."""
from __future__ import annotations

import json
import statistics

import torch
import torch.nn.functional as F

from .attention import Bolt3Attn, Bolt3Decoder
from .benchmark import _load_hf_donor, _load_rows, _load_student, gsm8k_prompt
from .profiles import resolve_profile
from .server import get_all_eos_ids

PARITY_EDGES = (0, 128, 256, 384, 1024, 2048, 3072)
DRIFT_EDGES = (0, 64, 128, 256, 512, 1024, 2048, 4096)


def bolt3_logits_fn(model):
    return lambda x: F.linear(model(x), model.lm_head.weight)


def hf_logits_fn(model):
    return lambda x: model(input_ids=x, use_cache=False).logits


def _bucketize(per_seq: list[torch.Tensor], edges=DRIFT_EDGES) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = [v[lo:hi] for v in per_seq if v.numel() > lo]
        n = sum(int(v.numel()) for v in vals)
        if n:
            rows.append({"pos": f"{lo}-{hi}", "mean": round(float(sum(v.sum() for v in vals)) / n, 4), "n": n})
    return rows


@torch.no_grad()
def score_continuation(s_fn, t_fn, seq: torch.Tensor, prompt_len: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L = seq.shape[1]
    with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
        ls = s_fn(seq)[0, prompt_len - 1 : L - 1]
        lt = t_fn(seq)[0, prompt_len - 1 : L - 1]
    tgt = seq[0, prompt_len:]
    kl, nll, agree = [], [], []
    for lo in range(0, tgt.numel(), 512):
        a = F.log_softmax(lt[lo : lo + 512].float(), dim=-1)
        b = F.log_softmax(ls[lo : lo + 512].float(), dim=-1)
        kl.append((a.exp() * (a - b)).sum(-1))
        nll.append(-a.gather(-1, tgt[lo : lo + 512, None])[:, 0])
        agree.append((a.argmax(-1) == b.argmax(-1)).float())
    del ls, lt
    return torch.cat(kl).cpu(), torch.cat(nll).cpu(), torch.cat(agree).cpu()


def _token_classes(tok, prompt_ids: list[int], gen_ids: list[int]) -> list[str]:
    def has_digit(t):
        return any(ch.isdigit() for ch in tok.decode([t]))

    prompt_digits = {t for t in prompt_ids if has_digit(t)}
    return [
        ("number (from question)" if t in prompt_digits else "number (new)") if has_digit(t) else "text"
        for t in gen_ids
    ]


@torch.no_grad()
def student_drift(
    stu,
    s_fn,
    t_fn,
    tok,
    device,
    prompts: list[str],
    max_new: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 1000,
) -> dict:
    stu.eval()
    dec = Bolt3Decoder(stu)
    eos_ids = get_all_eos_ids(tok)
    kls, nlls, lens = [], [], []
    for i, p in enumerate(prompts):
        torch.manual_seed(seed + i)
        x = torch.tensor([tok.encode(p, add_special_tokens=False)], dtype=torch.long, device=device)
        toks, n = dec.generate_batch(x, max_new, temperature=temperature, top_p=top_p, eos_ids=eos_ids)
        n = int(n[0])
        lens.append(n)
        kl, nll, _ = score_continuation(s_fn, t_fn, torch.cat([x, toks[:, :n]], dim=1), x.shape[1], device)
        kls.append(kl)
        nlls.append(nll)
    return {
        "kl": _bucketize(kls),
        "teacher_nll": _bucketize(nlls),
        "median_len": int(statistics.median(lens)) if lens else 0,
        "capped": f"{sum(int(n >= max_new) for n in lens)}/{len(lens)}",
        "capped_n": sum(int(n >= max_new) for n in lens),
        "n": len(lens),
    }


@torch.no_grad()
def teacher_forced_drift(
    teacher,
    s_fn,
    t_fn,
    tok,
    device,
    prompts: list[str],
    max_new: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 1000,
) -> dict:
    eos_ids = list(get_all_eos_ids(tok))
    kls, nlls, lens = [], [], []
    by_cls: dict[str, list[float]] = {}
    for i, p in enumerate(prompts):
        torch.manual_seed(seed + i)
        x = torch.tensor([tok.encode(p, add_special_tokens=False)], dtype=torch.long, device=device)
        out = teacher.generate(
            x,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_ids,
            pad_token_id=tok.eos_token_id or 0,
        )
        lens.append(out.shape[1] - x.shape[1])
        kl, nll, agree = score_continuation(s_fn, t_fn, out, x.shape[1], device)
        kls.append(kl)
        nlls.append(nll)
        classes = _token_classes(tok, x[0].tolist(), out[0, x.shape[1] :].tolist())
        for c, k_v, a_v in zip(classes, kl.tolist(), agree.tolist()):
            s = by_cls.setdefault(c, [0.0, 0.0, 0])
            s[0] += k_v
            s[1] += a_v
            s[2] += 1
    return {
        "kl": _bucketize(kls),
        "teacher_nll": _bucketize(nlls),
        "median_len": int(statistics.median(lens)) if lens else 0,
        "by_token_class": {
            c: {"kl": round(s[0] / s[2], 4), "top1_agree_pct": round(100.0 * s[1] / s[2], 2), "n": s[2]}
            for c, s in sorted(by_cls.items())
        },
    }


def _parity_tokens(tok, device, T: int) -> torch.Tensor:
    rows = _load_rows("openai/gsm8k", "main", "train", 400)
    ids = tok.encode(gsm8k_prompt(tok, rows[0]["question"]), add_special_tokens=False)
    for r in rows:
        if len(ids) >= T:
            break
        ids += tok.encode(f"{r['question'].strip()}\n{r['answer'].strip()}\n\n", add_special_tokens=False)
    T_use = min(T, len(ids))
    return torch.tensor([ids[:T_use]], dtype=torch.long, device=device)


@torch.no_grad()
def layer_parity(stu, tok, device, T: int = 1024) -> list[dict]:
    x = _parity_tokens(tok, device, T)
    T = x.shape[1]
    ins, outs = {}, {}

    def _hook(li):
        def fn(_m, args, out):
            ins[li], outs[li] = args[0].detach(), out.detach()
        return fn

    hooks = [blk.register_forward_hook(_hook(li)) for li, blk in enumerate(stu.blocks)]
    stu.eval()
    try:
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            stu(x)
    finally:
        for h in hooks:
            h.remove()

    dec = Bolt3Decoder(stu)
    states, kv = Bolt3Decoder.blank_state(stu, 1, device, max_kv_len=T + 64)
    edges = [e for e in (0, 128, 256, 384, 1024, 2048) if e < T] + [T]
    rows = []
    for li, blk in enumerate(stu.blocks):
        rec = torch.empty_like(outs[li])
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            for t in range(T):
                rec[:, t : t + 1] = dec.step_block(li, blk, ins[li][:, t : t + 1], t, states, kv)
        ref_d = (outs[li] - ins[li]).float()
        rec_d = (rec - ins[li]).float()
        err = ((rec_d - ref_d).norm(dim=-1) / ref_d.norm(dim=-1).clamp_min(1e-6))[0]
        row = {"layer": li, "type": "bolt3" if isinstance(blk.attn, Bolt3Attn) else "gdn"}
        for lo, hi in zip(edges[:-1], edges[1:]):
            row[f"{lo}-{hi}"] = round(float(err[lo:hi].mean()), 4)
        rows.append(row)
        del rec, ref_d, rec_d
    return rows


@torch.no_grad()
def decode_parity(stu, tok, device, T: int = 3072) -> list[dict]:
    x = _parity_tokens(tok, device, T)
    T = x.shape[1]

    stu.eval()
    with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
        par = F.linear(stu(x), stu.lm_head.weight)[0]
    dec = Bolt3Decoder(stu)
    states, kv = Bolt3Decoder.blank_state(stu, 1, device, max_kv_len=T + 64)
    max_diff, agree = torch.zeros(T), torch.zeros(T)
    for t in range(T):
        h, states, kv = dec.step(x[:, t : t + 1], t, states, kv)
        rec = stu.lm_head(h[:, -1])[0].float()
        ref = par[t].float()
        max_diff[t] = (rec - ref).abs().max().item()
        agree[t] = float(rec.argmax() == ref.argmax())

    out = []
    for lo, hi in zip(PARITY_EDGES[:-1], PARITY_EDGES[1:]):
        if lo >= T:
            break
        hi = min(hi, T)
        out.append(
            {
                "pos": f"{lo}-{hi}",
                "top1_agree_pct": round(100.0 * float(agree[lo:hi].mean()), 2),
                "max_abs_logit_diff": round(float(max_diff[lo:hi].max()), 3),
            }
        )
    return out


def _print_rows(title: str, rows: list[dict], key: str = "mean"):
    print(f"  {title}")
    for r in rows:
        print(f"    pos {r['pos']:>10}  {key}={r[key]}" + (f"  (n={r['n']})" if "n" in r else ""))


def run_diagnostics(
    ckpt_path: str,
    donor_id: str | None = None,
    n_prompts: int = 16,
    max_new: int | None = None,
    parity_T: int = 3072,
    out_json: str | None = None,
    parity_only: bool = False,
    skip_parity: bool = False,
    profile: str | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stu, _, tok, donor_id, donor_dir, step, _ = _load_student(ckpt_path, donor_id, device)
    prof = resolve_profile(tok, profile, donor_dir)
    max_new = prof.max_new if max_new is None else max_new
    res = {"checkpoint": ckpt_path, "step": step}

    print("=" * 84)
    print(f"BOLT-3 DIAGNOSTICS: {ckpt_path} (step {step})")
    print("=" * 84)
    if not skip_parity:
        res["decode_parity"] = decode_parity(stu, tok, device, T=parity_T)
        print("  [1] Decode parity (Bolt3Decoder token step vs chunked parallel forward; target >=99% top-1)")
        for r in res["decode_parity"]:
            flag = "ok" if r["top1_agree_pct"] >= 99.0 else "MISMATCH"
            print(
                f"    pos {r['pos']:>10}  top1 {r['top1_agree_pct']:6.2f}%  "
                f"max|dlogit| {r['max_abs_logit_diff']:.3f}  {flag}"
            )

        res["layer_parity"] = layer_parity(stu, tok, device, T=min(parity_T, 2048))
        print("  [1b] Per-layer parity (relative error per block)")
        for r in res["layer_parity"]:
            cols = "  ".join(f"{k}:{v:.4f}" for k, v in r.items() if k not in ("layer", "type"))
            print(f"    L{r['layer']:02d} {r['type']:>7}  {cols}")

    if parity_only:
        if out_json:
            with open(out_json, "w") as f:
                json.dump(res, f, indent=2)
        return res

    teacher = _load_hf_donor(donor_dir, device)
    s_fn, t_fn = bolt3_logits_fn(stu), hf_logits_fn(teacher)
    prompts = [gsm8k_prompt(tok, r["question"]) for r in _load_rows("openai/gsm8k", "main", "test", n_prompts)]

    res["teacher_forced"] = teacher_forced_drift(
        teacher, s_fn, t_fn, tok, device, prompts, max_new=max_new, temperature=prof.temperature, top_p=prof.top_p
    )
    res["free_run"] = student_drift(
        stu, s_fn, t_fn, tok, device, prompts, max_new=max_new, temperature=prof.temperature, top_p=prof.top_p
    )

    print(f"  [2] Teacher-forced (median len {res['teacher_forced']['median_len']})")
    _print_rows("KL(teacher||student) by position", res["teacher_forced"]["kl"])
    for c, v in res["teacher_forced"]["by_token_class"].items():
        print(f"    {c:>24}  KL {v['kl']:.4f}  top1 agree {v['top1_agree_pct']:6.2f}%  (n={v['n']})")
    print(f"  [3] Free-run (median len {res['free_run']['median_len']}, capped {res['free_run']['capped']})")
    _print_rows("KL(teacher||student) by position", res["free_run"]["kl"])
    _print_rows("teacher NLL by position", res["free_run"]["teacher_nll"])

    if out_json:
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Saved {out_json}")
    return res
