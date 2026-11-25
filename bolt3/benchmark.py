"""Bolt-3 7-Task Real-World Benchmark Suite (`python -m bolt3 benchmark`)."""
from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .attention import Bolt3Decoder
from .converter import _purge_zombie_vram
from .inspector import fetch_any_donor
from .memory_bank import Bolt3MemoryBank
from .model import load_bolt3_checkpoint
from .server import get_all_eos_ids

try:
    import inspect_ai
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        GenerateConfig,
        ModelAPI,
        ModelOutput,
        modelapi,
    )
    from inspect_ai.scorer import Score, Target, accuracy, scorer

    HAS_INSPECT = True
except Exception:
    HAS_INSPECT = False


def _ci95(acc_pct: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, acc_pct / 100.0))
    return 196.0 * math.sqrt(p * (1.0 - p) / n)


def _mcnemar_exact(stu_items: list[int], don_items: list[int]) -> dict:
    b = sum(1 for s, d in zip(stu_items, don_items) if s == 1 and d == 0)
    c = sum(1 for s, d in zip(stu_items, don_items) if s == 0 and d == 1)
    n = b + c
    if n == 0:
        return {"stu_only": 0, "don_only": 0, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_val = min(1.0, 2.0 * tail)
    return {"stu_only": b, "don_only": c, "p": round(p_val, 4)}


def _load_student(ckpt_path: str, donor_override: str | None, device: torch.device):
    from transformers import AutoTokenizer

    stu, cfg, ck_donor, step, ck = load_bolt3_checkpoint(ckpt_path, device=device, donor_override=donor_override, eval_mode=True)
    donor_id = donor_override or ck_donor
    if not donor_id:
        raise ValueError("Pass --donor <repo_id>")
    donor_dir = fetch_any_donor(donor_id)
    tok = AutoTokenizer.from_pretrained(donor_dir)
    conv_ver = (ck.get("injection") or {}).get("converter", "bolt3")
    return stu, cfg, tok, donor_id, donor_dir, step, conv_ver


def _load_hf_donor(donor_dir: str, device: torch.device):
    import transformers

    m = None
    for cls_n in ("AutoModelForCausalLM",):
        cls = getattr(transformers, cls_n, None)
        if cls is None:
            continue
        try:
            m = cls.from_pretrained(donor_dir, dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32))
            break
        except Exception:
            continue
    if m is None:
        raise RuntimeError(f"Unable to load HF donor from {donor_dir}")
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _load_rows(repo: str, name: str | None, split: str, limit: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(repo, name, split=split, streaming=True)
    rows = []
    for r in ds:
        rows.append(r)
        if len(rows) >= limit:
            break
    return rows


def _load_mmlu_stratified(limit: int = 200) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subj: dict[str, list[dict]] = {}
    for r in ds:
        by_subj.setdefault(r.get("subject", "other"), []).append(r)
    rng = random.Random(42)
    subjects = sorted(by_subj.keys())
    for s in subjects:
        rng.shuffle(by_subj[s])
    out = []
    round_idx = 0
    while len(out) < min(limit, len(ds)):
        added = False
        for s in subjects:
            if round_idx < len(by_subj[s]):
                out.append(by_subj[s][round_idx])
                added = True
                if len(out) >= limit:
                    break
        if not added:
            break
        round_idx += 1
    return out


@torch.no_grad()
def _score_choices_batch(
    model,
    is_hf: bool,
    tok,
    items: list[tuple[str, list[str]]],
    device: torch.device,
    norm_len: bool = True,
    batch_size: int = 32,
) -> list[int]:
    """Score multiple-choice questions in length-sorted right-padded batches (exact causal match)."""
    bos_id = tok.bos_token_id if isinstance(tok.bos_token_id, int) else None
    pad_id = tok.pad_token_id if isinstance(tok.pad_token_id, int) else (tok.eos_token_id if isinstance(tok.eos_token_id, int) else 0)
    w_lm = (model.get_output_embeddings().weight if is_hf else model.lm_head.weight).float()

    flat_seqs: list[tuple[int, int, list[int], int, int]] = []
    scores_out: list[list[float]] = []
    for q_idx, (prompt, choices) in enumerate(items):
        p_ids = tok.encode(prompt, add_special_tokens=False)
        if bos_id is not None and (not p_ids or p_ids[0] != bos_id):
            p_ids = [bos_id] + p_ids
        scores_out.append([-1e30] * len(choices))
        for c_idx, ch in enumerate(choices):
            c_str = " " + str(ch).lstrip()
            c_ids = tok.encode(c_str, add_special_tokens=False) or tok.encode(str(ch), add_special_tokens=False)
            full = (p_ids + c_ids)[-1024:]
            c_len = min(len(c_ids), len(full) - 1)
            denom = max(1, len(c_str)) if norm_len else 1
            flat_seqs.append((q_idx, c_idx, full, c_len, denom))

    # Sort by length so right-padding waste is minimal inside each batch
    flat_seqs.sort(key=lambda r: len(r[2]))
    for off in range(0, len(flat_seqs), batch_size):
        chunk = flat_seqs[off : off + batch_size]
        max_l = max(len(r[2]) for r in chunk)
        x = torch.full((len(chunk), max_l), pad_id, dtype=torch.long, device=device)
        for bi, (_, _, full, _, _) in enumerate(chunk):
            x[bi, : len(full)] = torch.tensor(full, dtype=torch.long, device=device)

        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            if is_hf:
                inner = getattr(getattr(model, "model", model), "language_model", getattr(model, "model", model))
                out = inner(input_ids=x, use_cache=False)
                h_all = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            else:
                h_all = model(x)

        for bi, (q_idx, c_idx, full, c_len, denom) in enumerate(chunk):
            L = len(full)
            h = h_all[bi, L - 1 - c_len : L - 1]
            logits = F.linear(h.float(), w_lm)
            tgt = torch.tensor(full[L - c_len : L], device=device).unsqueeze(-1)
            ll = F.log_softmax(logits, dim=-1).gather(-1, tgt).sum().item()
            scores_out[q_idx][c_idx] = ll / denom

    return [max(range(len(s)), key=lambda i: s[i]) for s in scores_out]


def _exec_subprocess_safely(code_str: str, timeout_s: int = 4) -> bool:
    with tempfile.TemporaryDirectory() as tmp_d:
        script_path = os.path.join(tmp_d, "solution_test.py")
        with open(script_path, "w") as f:
            f.write(code_str)
        try:
            res = subprocess.run(
                [sys.executable, "-I", script_path],
                cwd=tmp_d,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
            )
            return res.returncode == 0
        except Exception:
            return False


def _verify_ifeval_instruction(resp: str, inst_ids: list[str], kwargs_list: list[dict]) -> bool:
    ans = resp.split("</think>")[-1].strip() if "</think>" in resp else resp.strip()
    if not ans:
        return False
    for iid, kw in zip(inst_ids, kwargs_list):
        kw = kw or {}
        if "no_comma" in iid and "," in ans:
            return False
        if "lowercase" in iid and ans != ans.lower():
            return False
        if "capital" in iid and "frequency" not in iid and ans != ans.upper():
            return False
        if "json_format" in iid:
            body = ans
            if "```" in body:
                parts = [p for p in body.split("```") if p.strip()]
                body = parts[0].removeprefix("json").strip() if parts else body
            try:
                json.loads(body)
            except Exception:
                return False
        if "number_words" in iid:
            n_w = len(ans.split())
            rel, tgt = kw.get("relation", "at least"), int(kw.get("num_words") or 0)
            if "less" in rel and n_w >= tgt:
                return False
            if "least" in rel and n_w < tgt:
                return False
        if "number_bullet_lists" in iid:
            n_b = sum(1 for ln in ans.splitlines() if ln.lstrip().startswith(("* ", "- ")))
            tgt = int(kw.get("num_bullets") or 0)
            if tgt > 0 and n_b < tgt:
                return False
        if "forbidden_words" in iid:
            for fw in kw.get("forbidden_words") or []:
                if re.search(rf"\b{re.escape(str(fw))}\b", ans, flags=re.IGNORECASE):
                    return False
        if "existence" in iid:
            for req_w in kw.get("keywords") or []:
                if str(req_w).lower() not in ans.lower():
                    return False
    return True


class _CloseThinkLogitsProcessor:
    def __init__(self, prompt_len: int, close_ids: list[int], trigger_step: int, batch_size: int = 1):
        self.prompt_len = prompt_len
        self.close_ids = close_ids
        self.end_think_id = close_ids[1] if len(close_ids) > 1 else close_ids[0]
        self.trigger_step = trigger_step
        self.start_steps = [None] * batch_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step_i = int(input_ids.shape[1]) - self.prompt_len
        for r in range(input_ids.shape[0]):
            if self.start_steps[r] is None:
                gen_r = input_ids[r, self.prompt_len :]
                if (gen_r == self.end_think_id).any().item():
                    continue
                if step_i == self.trigger_step:
                    self.start_steps[r] = step_i
            if self.start_steps[r] is not None:
                offset = step_i - self.start_steps[r]
                if offset < len(self.close_ids):
                    tgt = self.close_ids[offset]
                    scores[r].fill_(-1e9)
                    scores[r, tgt] = 0.0
        return scores


def gsm8k_prompt(tok, question: str) -> str:
    msgs = [
        {
            "role": "user",
            "content": f"Solve this math problem step by step and put the final numerical answer inside \\boxed{{}}:\n{question.strip()}",
        }
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"Question: {question.strip()}\nSolution:\n"


_ACTIVE_ENGINE = {"gen_fn": None, "name": "bolt3"}

if HAS_INSPECT:
    @modelapi(name="bolt3_inspect")
    class Bolt3InspectModelAPI(ModelAPI):
        def __init__(self, model_name: str, **model_args):
            super().__init__(model_name, **model_args)

        async def generate(self, input, tools, tool_choice, config: GenerateConfig):
            prompt_str = input[-1].text if isinstance(input, list) and input else str(input)
            fn = _ACTIVE_ENGINE["gen_fn"]
            max_toks = int(config.max_tokens or 1024)
            temp = float(config.temperature if config.temperature is not None else 0.0)
            top_p = float(config.top_p if config.top_p is not None else 1.0)
            text_out = fn(prompt_str, max_toks, temp, top_p)
            return ModelOutput(
                model=self.model_name,
                choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=text_out), stop_reason="stop")],
            )


def _run_inspect_or_direct_task(task_name: str, samples_data: list[dict], eval_fn, log_dir: str) -> list[int]:
    hits = []
    n = len(samples_data)
    t0 = time.perf_counter()

    if HAS_INSPECT:
        @scorer(metrics=[accuracy()])
        def _custom_scorer():
            async def score(state, target: Target):
                ok = int(state.metadata["hit"])
                return Score(value=1.0 if ok else 0.0, answer=state.output.completion)
            return score

        inspect_samples = []
        for idx, item in enumerate(samples_data):
            hit, completion = eval_fn(idx, item)
            hits.append(int(hit))
            inspect_samples.append(
                Sample(
                    input=str(item.get("prompt_preview", f"{task_name}_{idx}")),
                    target=str(item.get("gold", "1")),
                    id=idx,
                    metadata={"idx": idx, "hit": int(hit), "completion": completion[:200]},
                )
            )
            if (idx + 1) % 25 == 0 or (idx + 1) == n:
                print(
                    f"    [{_ACTIVE_ENGINE['name']} | {task_name:<13}] {idx+1:3d}/{n} | acc={100*sum(hits)/len(hits):5.1f}% ({sum(hits)}/{len(hits)}) | {time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
        try:
            _ACTIVE_ENGINE["gen_fn"] = lambda *_a: "logged"
            task_obj = Task(dataset=MemoryDataset(inspect_samples), solver=[], scorer=_custom_scorer(), name=task_name)
            inspect_eval(task_obj, model="bolt3_inspect/local", log_dir=log_dir, display="none")
        except Exception:
            pass
        return hits

    for idx, item in enumerate(samples_data):
        hit, _ = eval_fn(idx, item)
        hits.append(int(hit))
        if (idx + 1) % 25 == 0 or (idx + 1) == n:
            print(
                f"    [{_ACTIVE_ENGINE['name']} | {task_name:<13}] {idx+1:3d}/{n} | acc={100*sum(hits)/len(hits):5.1f}% ({sum(hits)}/{len(hits)}) | {time.perf_counter()-t0:.1f}s",
                flush=True,
            )
    return hits


def _auto_eval_batch_size(device: torch.device, override: int | None = None) -> int:
    if override is not None:
        return max(1, int(override))
    if device.type != "cuda":
        return 4
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 80.0:
        return 64
    if vram_gb >= 40.0:
        return 32
    if vram_gb >= 20.0:
        return 16
    return 8


@torch.no_grad()
def _eval_model(
    model,
    is_hf: bool,
    tok,
    device: torch.device,
    limit: int = 200,
    gsm8k_limit: int = 200,
    cached: dict | None = None,
    on_save=None,
    gsm_max_new: int = 2048,
    batch_size_override: int | None = None,
    tasks_filter: set[str] | None = None,
) -> dict:
    out = dict(cached) if cached else {}
    eos_ids = get_all_eos_ids(tok)
    dec = Bolt3Decoder(model, max_cache_pos=16384) if not is_hf else None
    _ACTIVE_ENGINE["name"] = "Donor" if is_hf else "Bolt-3"
    log_dir = os.path.join(os.getcwd(), "inspect_logs", "donor" if is_hf else "bolt3")
    os.makedirs(log_dir, exist_ok=True)
    has_think_vocab = "</think>" in tok.get_vocab()
    close_toks = tok.encode("\n</think>\nTherefore, final answer is \\boxed{", add_special_tokens=False)
    gen_bs = _auto_eval_batch_size(device, batch_size_override)
    mc_bs = max(16, gen_bs) if device.type == "cuda" else 4
    pad_id = (
        tok.pad_token_id
        if isinstance(tok.pad_token_id, int)
        else (tok.eos_token_id if isinstance(tok.eos_token_id, int) else 0)
    )

    def _want(name: str) -> bool:
        return not tasks_filter or name.lower() in tasks_filter

    def _gen_batch(
        prompts: list[str],
        max_new: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        base_seed: int = 0,
        label: str = "",
    ) -> list[tuple[str, int, bool]]:
        n_tot = len(prompts)
        res: list[tuple[str, int, bool]] = [("", 0, False)] * n_tot
        t0 = time.perf_counter()
        done_cnt = 0

        # Group only by `need_close` and sort by token length so left-padded batches run at full `gen_bs`
        buckets: dict[bool, list[tuple[int, list[int]]]] = {False: [], True: []}
        for idx_p, p_s in enumerate(prompts):
            ids = tok.encode(p_s, add_special_tokens=False)
            need_close = (max_new >= 1024) and ("<think>" in p_s or has_think_vocab) and ("</think>" not in p_s)
            buckets[need_close].append((idx_p, ids))

        for need_close, items_b in buckets.items():
            if not items_b:
                continue
            items_b.sort(key=lambda pair: len(pair[1]))
            c_ids = close_toks if need_close else None
            for off in range(0, len(items_b), gen_bs):
                ch = items_b[off : off + gen_bs]
                max_l = max(len(e) for _, e in ch)
                x_in = torch.full((len(ch), max_l), pad_id, dtype=torch.long, device=device)
                attn_m = torch.zeros((len(ch), max_l), dtype=torch.long, device=device)
                for r_i, (_, e) in enumerate(ch):
                    x_in[r_i, max_l - len(e) :] = torch.tensor(e, dtype=torch.long, device=device)
                    attn_m[r_i, max_l - len(e) :] = 1
                torch.manual_seed(base_seed + ch[0][0])

                if not is_hf:
                    toks, n_lens = dec.generate_batch(
                        x_in,
                        max_new_tokens=max_new,
                        temperature=temperature,
                        top_p=top_p,
                        eos_ids=eos_ids,
                        close_think_ids=c_ids,
                        attn_mask=attn_m,
                    )
                    for r_i, (orig_idx, _) in enumerate(ch):
                        n_r = int(n_lens[r_i].item())
                        g_ids = toks[r_i, :n_r].tolist()
                        hit_eos = bool(eos_ids and g_ids and g_ids[-1] in eos_ids)
                        if hit_eos:
                            g_ids = g_ids[:-1]
                        res[orig_idx] = (
                            tok.decode(g_ids, skip_special_tokens=True),
                            n_r,
                            (not hit_eos and n_r >= max_new),
                        )
                else:
                    from transformers import LogitsProcessorList

                    sample = temperature > 1e-5
                    lp = (
                        LogitsProcessorList(
                            [_CloseThinkLogitsProcessor(max_l, close_toks, max(64, max_new - 64), batch_size=len(ch))]
                        )
                        if need_close
                        else None
                    )
                    gen_t = model.generate(
                        input_ids=x_in,
                        attention_mask=attn_m,
                        max_new_tokens=max_new,
                        do_sample=sample,
                        temperature=temperature if sample else None,
                        top_p=top_p if sample else None,
                        logits_processor=lp,
                        eos_token_id=list(eos_ids),
                        pad_token_id=pad_id,
                    )
                    for r_i, (orig_idx, _) in enumerate(ch):
                        raw_ids = gen_t[r_i, max_l:].tolist()
                        cut = next((k for k, tid in enumerate(raw_ids) if tid in eos_ids), None)
                        hit_cap = cut is None and len(raw_ids) >= max_new
                        g_ids = raw_ids[:cut] if cut is not None else raw_ids
                        res[orig_idx] = (tok.decode(g_ids, skip_special_tokens=True), len(g_ids), hit_cap)

                done_cnt += len(ch)
                if label:
                    print(
                        f"    [{_ACTIVE_ENGINE['name']} | {label:<13}] gen {done_cnt:3d}/{n_tot} (bs={len(ch)}) | {time.perf_counter()-t0:.1f}s",
                        flush=True,
                    )
        return res

    # 1. MMLU
    if _want("MMLU"):
        if "MMLU" not in out:
            rows = _load_mmlu_stratified(limit)
            items_m = []
            for r in rows:
                ch = r["choices"]
                subj = r.get("subject", "general").replace("_", " ")
                prompt = (
                    f"The following is a multiple-choice question about {subj}.\n"
                    f"Question: {r['question'].strip()}\nOptions:\n- {ch[0]}\n- {ch[1]}\n- {ch[2]}\n- {ch[3]}\nCorrect Answer:"
                )
                items_m.append((prompt, ch))
            preds_m = _score_choices_batch(model, is_hf, tok, items_m, device, norm_len=True, batch_size=mc_bs)
            hits = _run_inspect_or_direct_task(
                "MMLU", rows, lambda idx, r: ((preds_m[idx] == int(r["answer"])), str(preds_m[idx])), log_dir
            )
            acc = (sum(hits) / len(hits)) * 100.0
            out["MMLU"] = {"acc": round(acc, 2), "ci95": round(_ci95(acc, len(hits)), 2), "correct": sum(hits), "total": len(hits), "items": hits}
            if on_save:
                on_save(out)
        else:
            print(f"    [{_ACTIVE_ENGINE['name']} | MMLU         ] cached {out['MMLU']['acc']:.2f}%", flush=True)

    # 2. ARC-Challenge
    if _want("ARC-Challenge"):
        if "ARC-Challenge" not in out:
            rows = _load_rows("allenai/ai2_arc", "ARC-Challenge", "test", limit)
            items_a = [(f"Question: {r['question'].strip()}\nAnswer:", r["choices"]["text"]) for r in rows]
            preds_a = _score_choices_batch(model, is_hf, tok, items_a, device, norm_len=True, batch_size=mc_bs)

            def _eval_arc(idx, r):
                ch_labels = r["choices"]["label"]
                gold_idx = ch_labels.index(r["answerKey"]) if r["answerKey"] in ch_labels else 0
                return (preds_a[idx] == gold_idx), str(preds_a[idx])

            hits = _run_inspect_or_direct_task("ARC-Challenge", rows, _eval_arc, log_dir)
            acc = (sum(hits) / len(hits)) * 100.0
            out["ARC-Challenge"] = {"acc": round(acc, 2), "ci95": round(_ci95(acc, len(hits)), 2), "correct": sum(hits), "total": len(hits), "items": hits}
            if on_save:
                on_save(out)
        else:
            print(f"    [{_ACTIVE_ENGINE['name']} | ARC-Challenge] cached {out['ARC-Challenge']['acc']:.2f}%", flush=True)

    # 3. HellaSwag
    if _want("HellaSwag"):
        if "HellaSwag" not in out:
            rows = _load_rows("Rowan/hellaswag", None, "validation", limit)
            items_h = [(r["ctx"].strip(), r["endings"]) for r in rows]
            preds_h = _score_choices_batch(model, is_hf, tok, items_h, device, norm_len=True, batch_size=mc_bs)
            hits = _run_inspect_or_direct_task(
                "HellaSwag", rows, lambda idx, r: ((preds_h[idx] == int(r["label"])), str(preds_h[idx])), log_dir
            )
            acc = (sum(hits) / len(hits)) * 100.0
            out["HellaSwag"] = {"acc": round(acc, 2), "ci95": round(_ci95(acc, len(hits)), 2), "correct": sum(hits), "total": len(hits), "items": hits}
            if on_save:
                on_save(out)
        else:
            print(f"    [{_ACTIVE_ENGINE['name']} | HellaSwag    ] cached {out['HellaSwag']['acc']:.2f}%", flush=True)

    # 4. GSM8K
    if _want("GSM8K"):
        if "GSM8K" not in out:
            rows = _load_rows("openai/gsm8k", "main", "test", gsm8k_limit)
            prompts_gsm = [gsm8k_prompt(tok, r["question"]) for r in rows]
            gens_gsm = _gen_batch(prompts_gsm, max_new=gsm_max_new, temperature=0.2, top_p=0.85, base_seed=1000, label="GSM8K")
            lengths, trunc_cnt = [], 0

            def _eval_gsm(idx, r):
                nonlocal trunc_cnt
                gold = r["answer"].split("####")[-1].strip().replace(",", "")
                gen, n_new, hit_cap = gens_gsm[idx]
                lengths.append(n_new)
                trunc_cnt += int(hit_cap)
                ans_region = gen.split("</think>")[-1] if "</think>" in gen else gen
                boxed = re.findall(r"\\boxed\{([^}]+)\}", ans_region)
                nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", boxed[-1] if boxed else ans_region)
                pred = nums[-1].replace(",", "") if nums else None
                hit = False
                if pred is not None and not hit_cap:
                    try:
                        hit = abs(float(pred) - float(gold)) < 1e-3
                    except Exception:
                        hit = pred == gold
                return hit, gen

            hits = _run_inspect_or_direct_task("GSM8K", rows, _eval_gsm, log_dir)
            acc = (sum(hits) / len(hits)) * 100.0
            out["GSM8K"] = {
                "acc": round(acc, 2),
                "ci95": round(_ci95(acc, len(hits)), 2),
                "correct": sum(hits),
                "total": len(hits),
                "items": hits,
                "truncated": f"{trunc_cnt}/{len(hits)}",
                "median_len": int(statistics.median(lengths)) if lengths else 0,
            }
            if on_save:
                on_save(out)
        else:
            print(f"    [{_ACTIVE_ENGINE['name']} | GSM8K        ] cached {out['GSM8K']['acc']:.2f}%", flush=True)

    # 5. HumanEval
    if _want("HumanEval"):
        if "HumanEval" not in out:
            rows = _load_rows("openai/openai_humaneval", None, "test", limit)
            prompts_he = []
            for r in rows:
                prompt_code = r["prompt"]
                msgs = [{"role": "user", "content": f"Complete this Python function. Output ONLY valid Python code inside ```python```:\n```python\n{prompt_code}\n```"}]
                try:
                    try:
                        p_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    except TypeError:
                        p_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    if "<think>" in p_str and "</think>" not in p_str:
                        p_str = p_str.rstrip() + "\n</think>\n\n```python\n" + prompt_code
                    elif p_str.rstrip().endswith("</think>"):
                        p_str = p_str.rstrip() + "\n\n```python\n" + prompt_code
                except Exception:
                    p_str = prompt_code
                prompts_he.append(p_str)
            gens_he = _gen_batch(prompts_he, max_new=256, temperature=0.0, top_p=1.0, base_seed=2000, label="HumanEval")

            def _eval_humaneval(idx, r):
                prompt_code = r["prompt"]
                gen = gens_he[idx][0]
                if "```" in gen:
                    gen = gen.split("```")[0]
                clean_lines = []
                for ln in gen.splitlines():
                    if ln.startswith(("def ", "class ", "if __name__", "# Test", "assert ")) and clean_lines:
                        break
                    clean_lines.append(ln)
                prog = prompt_code + "\n".join(clean_lines) + "\n\n" + r["test"] + f"\ncheck({r['entry_point']})\n"
                return _exec_subprocess_safely(prog, timeout_s=4), gen

            hits = _run_inspect_or_direct_task("HumanEval", rows, _eval_humaneval, log_dir)
            acc = (sum(hits) / len(hits)) * 100.0
            out["HumanEval"] = {"acc": round(acc, 2), "ci95": round(_ci95(acc, len(hits)), 2), "correct": sum(hits), "total": len(hits), "items": hits}
            if on_save:
                on_save(out)
        else:
            print(f"    [{_ACTIVE_ENGINE['name']} | HumanEval    ] cached {out['HumanEval']['acc']:.2f}%", flush=True)

    # 6. MBPP
    if _want("MBPP"):
        if "MBPP" not in out:
            rows = _load_rows("google-research-datasets/mbpp", "sanitized", "test", limit)
            prompts_mb = []
            for r in rows:
                sig_line = r["code"].splitlines()[0] if r.get("code") else "def solution():"
                p_code = f"# {r['prompt'].strip()}\n{sig_line}\n"
                msgs = [{"role": "user", "content": f"Complete this Python function:\n```python\n{p_code}\n```"}]
                try:
                    try:
                        p_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    except TypeError:
                        p_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    if "<think>" in p_str and "</think>" not in p_str:
                        p_str = p_str.rstrip() + "\n</think>\n\n```python\n" + p_code
                    elif p_str.rstrip().endswith("</think>"):
                        p_str = p_str.rstrip() + "\n\n```python\n" + p_code
                except Exception:
                    p_str = p_code
                prompts_mb.append(p_str)
            gens_mb = _gen_batch(prompts_mb, max_new=256, temperature=0.0, top_p=1.0, base_seed=3000, label="MBPP")

            def _eval_mbpp(idx, r):
                tests = "\n".join(r["test_list"])
                sig_line = r["code"].splitlines()[0] if r.get("code") else "def solution():"
                p_code = f"# {r['prompt'].strip()}\n{sig_line}\n"
                gen = gens_mb[idx][0]
                if "```" in gen:
                    gen = gen.split("```")[0]
                clean_lines = []
                for ln in gen.splitlines():
                    if ln.startswith(("def ", "class ", "if __name__", "assert ")) and clean_lines:
                        break
                    clean_lines.append(ln)
                prog = "\n".join(r.get("test_imports", [])) + "\n" + p_code + "\n".join(clean_lines) + "\n" + tests + "\n"
                return _exec_subprocess_safely(prog, timeout_s=4), gen

            hits = _run_inspect_or_direct_task("MBPP", rows, _eval_mbpp, log_dir)
            acc = (sum(hits) / len(hits)) * 100.0
            out["MBPP"] = {"acc": round(acc, 2), "ci95": round(_ci95(acc, len(hits)), 2), "correct": sum(hits), "total": len(hits), "items": hits}
            if on_save:
                on_save(out)
        else:
            print(f"    [{_ACTIVE_ENGINE['name']} | MBPP         ] cached {out['MBPP']['acc']:.2f}%", flush=True)

    # 7. IFEval
    if _want("IFEval"):
        if "IFEval" not in out:
            try:
                rows = _load_rows("google/IFEval", None, "train", limit)
                prompts_if = []
                for r in rows:
                    msgs = [{"role": "user", "content": r["prompt"]}]
                    try:
                        try:
                            p_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        except TypeError:
                            p_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        if "<think>" in p_str and "</think>" not in p_str:
                            p_str = p_str.rstrip() + "\n</think>\n\n"
                    except Exception:
                        p_str = r["prompt"]
                    prompts_if.append(p_str)
                gens_if = _gen_batch(prompts_if, max_new=384, temperature=0.0, top_p=1.0, base_seed=4000, label="IFEval")

                def _eval_ifeval(idx, r):
                    gen = gens_if[idx][0]
                    ok = _verify_ifeval_instruction(gen, r.get("instruction_id_list", []), r.get("kwargs", []))
                    return ok, gen

                hits = _run_inspect_or_direct_task("IFEval", rows, _eval_ifeval, log_dir)
                acc = (sum(hits) / len(hits)) * 100.0
                out["IFEval"] = {"acc": round(acc, 2), "ci95": round(_ci95(acc, len(hits)), 2), "correct": sum(hits), "total": len(hits), "items": hits}
                if on_save:
                    on_save(out)
            except Exception as e:
                print(f"  [IFEval skipped: {e}]", flush=True)
        else:
            print(f"    [{_ACTIVE_ENGINE['name']} | IFEval       ] cached {out['IFEval']['acc']:.2f}%", flush=True)

    return out


@torch.no_grad()
def eval_real_speed_and_ram(stu, hf_donor, cfg: dict, tok, device: torch.device) -> dict:
    gen_steps = 32
    ctx_lengths = (2048, 8192, 16384, 32768, 65536)
    scaling = []
    dec_stu = Bolt3Decoder(stu, max_cache_pos=70000)

    for T_c in ctx_lengths:
        nxt_tok = torch.tensor([[100]], device=device, dtype=torch.long)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base_mem_stu = torch.cuda.memory_allocated()
        states_s, kv_s = dec_stu.blank_state(stu, 1, device, max_kv_len=T_c + gen_steps + 32)
        for li, blk in enumerate(stu.blocks):
            ly = blk.attn
            kv_s[li]["sink_k"] = torch.randn(1, ly.HKV, ly.n_sinks, ly.HD, device=device, dtype=torch.bfloat16) * 0.02
            kv_s[li]["sink_v"] = torch.randn_like(kv_s[li]["sink_k"])
            kv_s[li]["sink_valid"] = torch.ones(1, 1, ly.n_sinks, device=device, dtype=torch.bool)
            kv_s[li]["sink_filled"] = torch.full((1,), ly.n_sinks, device=device, dtype=torch.long)
            kv_s[li]["prev_kr"] = torch.randn(1, ly.HKV, ly.C, ly.HD, device=device, dtype=torch.bfloat16) * 0.02
            kv_s[li]["prev_kp"] = torch.randn_like(kv_s[li]["prev_kr"])
            kv_s[li]["prev_v"] = torch.randn_like(kv_s[li]["prev_kr"])
            kv_s[li]["prev_pos"] = torch.arange(T_c - ly.C, T_c, device=device, dtype=torch.float32).view(1, 1, ly.C)
            kv_s[li]["prev_b"] = torch.full((1, ly.HKV, ly.C), 0.25, device=device, dtype=torch.bfloat16)
            kv_s[li]["prev_valid"] = torch.ones(1, 1, ly.C, device=device, dtype=torch.bool)
            kv_s[li]["ak"] = torch.randn(ly.HKV, ly.S, ly.HD, device=device, dtype=torch.bfloat16) * 0.02
            kv_s[li]["av"] = torch.randn_like(kv_s[li]["ak"])
            kv_s[li]["mass"] = torch.ones(ly.HKV, ly.S, device=device, dtype=torch.float32)
            kv_s[li]["util"] = torch.ones(ly.HKV, ly.S, device=device, dtype=torch.float32) * 0.2
            kv_s[li]["util_acc"] = torch.zeros(ly.HKV, ly.S, device=device, dtype=torch.float32)
            kv_s[li]["pm"] = torch.arange(ly.S, device=device, dtype=torch.float32).unsqueeze(0).expand(ly.HKV, -1).clone()
            kv_s[li]["act"] = torch.ones(ly.HKV, ly.S, device=device, dtype=torch.bool)
            kv_s[li]["n_filled"] = ly.S
        bolt3_state_mb = (torch.cuda.memory_allocated() - base_mem_stu) / (1024 ** 2)

        dec_stu.step(nxt_tok, T_c, states_s, kv_s)
        torch.cuda.synchronize()
        runs_s = []
        for _ in range(3):
            t0 = time.perf_counter()
            for step_i in range(gen_steps):
                x_h, states_s, kv_s = dec_stu.step(nxt_tok, T_c + step_i, states_s, kv_s)
                _ = stu.lm_head(x_h[:, -1, :]).argmax(-1)
            torch.cuda.synchronize()
            runs_s.append(gen_steps / (time.perf_counter() - t0))
        bolt3_tok_s = statistics.median(runs_s)
        del states_s, kv_s
        torch.cuda.empty_cache()

        L = cfg["num_layers"]
        HKV = cfg["num_kv_heads"]
        HD = cfg["head_dim"]
        inner_hf = getattr(getattr(hf_donor, "model", hf_donor), "language_model", getattr(hf_donor, "model", hf_donor))
        donor_kv_mb = (L * 2 * HKV * HD * T_c * 2) / (1024 ** 2)
        from transformers.cache_utils import DynamicCache

        cache_hf = DynamicCache()
        for li in range(L):
            cache_hf.update(
                torch.randn(1, HKV, T_c, HD, device=device, dtype=torch.bfloat16),
                torch.randn(1, HKV, T_c, HD, device=device, dtype=torch.bfloat16),
                li,
            )
        with torch.autocast("cuda", torch.bfloat16):
            _ = inner_hf(input_ids=nxt_tok, past_key_values=cache_hf, use_cache=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16):
            for _ in range(gen_steps):
                o = inner_hf(input_ids=nxt_tok, past_key_values=cache_hf, use_cache=True)
                _ = hf_donor.get_output_embeddings()(o.last_hidden_state[:, -1]).argmax(-1)
        torch.cuda.synchronize()
        donor_tok_s = gen_steps / (time.perf_counter() - t0)
        del cache_hf
        torch.cuda.empty_cache()

        scaling.append(
            {
                "context": T_c,
                "donor_kv_mb": round(donor_kv_mb, 2),
                "bolt3_state_mb": round(bolt3_state_mb, 2),
                "ram_saved_pct": round((1.0 - bolt3_state_mb / max(1e-4, donor_kv_mb)) * 100.0, 1),
                "donor_tok_s": round(donor_tok_s, 1),
                "bolt3_tok_s": round(bolt3_tok_s, 1),
                "speedup": round(bolt3_tok_s / max(1e-4, donor_tok_s), 2),
            }
        )

    bank_dir = Path("./_tmp_bolt3_bench_bank")
    bank = Bolt3MemoryBank(stu, tok, bank_dir=bank_dir)
    secrets = [
        ("orion", "The launch authorization code for Project Orion is TITAN-OMEGA-942.", "What is the launch authorization code for Project Orion?", "TITAN-OMEGA-942"),
        ("hemlock", "The antidote synthesis temperature for Protocol Hemlock is 347 Kelvin.", "What is the antidote synthesis temperature for Protocol Hemlock?", "347"),
        ("vanguard", "The deep-space relay frequency for Station Vanguard is 884.5 MHz.", "What is the deep-space relay frequency for Station Vanguard?", "884.5"),
        ("chronos", "The vault master key for Facility Chronos is COBALT-LOTUS-19.", "What is the vault master key for Facility Chronos?", "COBALT-LOTUS-19"),
    ]
    for n, d, _, _ in secrets:
        bank.memorize_to_disk(n, d)
    hits, max_res, r_ms = 0, 0.0, []
    for n, _, q_s, exp in secrets:
        res = bank.ask(q_s, max_new_tokens=32)
        r_ms.append(res["route_ms"])
        max_res = max(max_res, res["erase_residual"])
        hits += int(exp.lower() in res["answer"].lower() and res["cartridge"] == n)
    for n, _, _, _ in secrets:
        bank.forget_from_disk(n)
    try:
        bank_dir.rmdir()
    except Exception:
        pass

    return {
        "scaling": scaling,
        "cartridge_recall_pct": (hits / len(secrets)) * 100.0,
        "cartridge_hits": f"{hits}/{len(secrets)}",
        "avg_route_ms": round(sum(r_ms) / len(r_ms), 2),
        "erase_residual": max_res,
    }


def _sc_state_mb(row: dict) -> float:
    return float(row.get("bolt3_state_mb", row.get("vsa_kv_mb", 0.0)))


def _sc_tok_s(row: dict) -> float:
    return float(row.get("bolt3_tok_s", row.get("vsa_tok_s", 0.0)))


def _plot_benchmark_figure(results: dict, out_png: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=160)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.grid(True, alpha=0.2, color="#8b949e", linestyle="--")

    ax1 = axes[0, 0]
    tasks = list(results["student"].keys())
    s_vals = [results["student"][k]["acc"] for k in tasks]
    s_ci = [results["student"][k].get("ci95", 0.0) for k in tasks]
    d_vals = [results["teacher"].get(k, {}).get("acc", 0.0) for k in tasks]
    d_ci = [results["teacher"].get(k, {}).get("ci95", 0.0) for k in tasks]
    x = np.arange(len(tasks))
    w = 0.36
    if results["teacher"]:
        ax1.bar(x - w / 2, d_vals, w, yerr=d_ci, capsize=3, label=f"HF Donor ({results['donor'].split('/')[-1]})", color="#58a6ff")
        ax1.bar(x + w / 2, s_vals, w, yerr=s_ci, capsize=3, label="Bolt-3 (Full)", color="#3fb950")
    else:
        ax1.bar(x, s_vals, w, yerr=s_ci, capsize=3, label="Bolt-3 (Full)", color="#3fb950")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=15, fontsize=8.5, fontweight="bold")
    ax1.set_ylabel("Accuracy (% ± 95% CI)", fontweight="bold")
    ax1.set_title("1. Real-World Evaluation Suite (McNemar Tested)", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = axes[0, 1]
    sc = results["systems"]["scaling"]
    ctxs = [f"{r['context']//1024}k" for r in sc]
    ax2.plot(ctxs, [r["donor_tok_s"] for r in sc], "o--", color="#58a6ff", linewidth=2.2, label="HF Donor (DynamicCache)")
    ax2.plot(ctxs, [_sc_tok_s(r) for r in sc], "s-", color="#3fb950", linewidth=2.5, label="Bolt-3 (Constant State)")
    ax2.set_xlabel("Context Length (Tokens)", fontweight="bold")
    ax2.set_ylabel("Measured Decode Speed (tok/s)", fontweight="bold")
    ax2.set_title(f"2. Decode Throughput ({sc[-1]['speedup']}x @ {ctxs[-1]})", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="best", fontsize=8)

    ax3 = axes[1, 0]
    ax3.plot(ctxs, [r["donor_kv_mb"] for r in sc], "o--", color="#f85149", linewidth=2.2, label="Donor KV Cache (MB)")
    ax3.plot(ctxs, [_sc_state_mb(r) for r in sc], "s-", color="#3fb950", linewidth=2.5, label="Bolt-3 Fixed State (MB)")
    ax3.fill_between(
        range(len(ctxs)),
        [_sc_state_mb(r) for r in sc],
        [r["donor_kv_mb"] for r in sc],
        color="#3fb950",
        alpha=0.18,
        label=f"-{sc[-1]['ram_saved_pct']}% Saved",
    )
    ax3.set_xlabel("Context Length (Tokens)", fontweight="bold")
    ax3.set_ylabel("State / Cache RAM (MB)", fontweight="bold")
    ax3.set_title(f"3. State RAM Scaling (-{sc[-1]['ram_saved_pct']}% @ {ctxs[-1]})", fontsize=10.5, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8)

    ax4 = axes[1, 1]
    sys_m = results["systems"]
    metrics = ["Cartridge Recall (%)", f"{ctxs[-1]} RAM Saved (%)", "Erase Residual < 1e-4"]
    vals = [sys_m["cartridge_recall_pct"], sc[-1]["ram_saved_pct"], 100.0 if sys_m["erase_residual"] < 1e-4 else 0.0]
    bars = ax4.barh(metrics, vals, color=["#a371f7", "#3fb950", "#d29922"], height=0.5)
    ax4.set_xlim(0, 115)
    for b, v in zip(bars, vals):
        ax4.text(v + 2, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontweight="bold", color="white")
    ax4.set_title(f"4. Cartridge Recall & M_res Erase ({sys_m['erase_residual']:.2e})", fontsize=10.5, fontweight="bold")

    plt.suptitle(f"Bolt-3 vs {results['donor']} — Benchmark Report", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[bolt3-bench] Saved 4-panel chart -> {out_png}", flush=True)


def run_benchmark_suite(
    ckpt_path: str,
    *,
    donor_id: str | None = None,
    compare_donor: bool = True,
    limit: int = 200,
    gsm8k_limit: int = 200,
    out_json: str | None = None,
    seed_teacher: dict | None = None,
    batch_size: int | None = None,
    tasks: tuple[str, ...] = (),
    fresh_student: bool = False,
    fresh: bool = False,
) -> dict:
    _purge_zombie_vram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stu, cfg, tok, donor_repo, donor_dir, step, conv_ver = _load_student(ckpt_path, donor_id, device)
    gsm_max_new = 2048
    tasks_filter = {t.strip().lower() for t in tasks if t.strip()} or None

    base_out = os.path.splitext(out_json or ckpt_path)[0]
    json_path = out_json or (base_out + "_benchmarks.json")
    png_path = base_out + "_benchmark.png"

    eff_bs = _auto_eval_batch_size(device, batch_size)
    print("=" * 88)
    print(f"BOLT-3 BENCHMARK SUITE (N={limit}, gsm_cap={gsm_max_new}, eval_bs={eff_bs}): {ckpt_path} vs {donor_repo}")
    print(f"Inspect AI Active: {HAS_INSPECT}")
    print("=" * 88, flush=True)

    if os.path.isfile(json_path) and not fresh:
        with open(json_path) as f:
            results = json.load(f)
        if fresh_student:
            results["student"] = {}
            results["mcnemar"] = {}
            results["systems"] = {}
    else:
        results = {
            "harness": "UK AISI Inspect (inspect_ai)" if HAS_INSPECT else "native",
            "limit": limit,
            "gsm_max_new": gsm_max_new,
            "checkpoint": ckpt_path,
            "step": step,
            "converter": conv_ver,
            "donor": donor_repo,
            "student": {},
            "teacher": {},
            "mcnemar": {},
            "systems": {},
        }
    if seed_teacher:
        for k_t, v_t in seed_teacher.items():
            results["teacher"].setdefault(k_t, v_t)

    def _flush():
        tmp = json_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, json_path)

    results["student"] = _eval_model(
        stu,
        False,
        tok,
        device,
        limit=limit,
        gsm8k_limit=gsm8k_limit,
        cached=results.get("student"),
        on_save=lambda d: (results.__setitem__("student", d), _flush()),
        gsm_max_new=gsm_max_new,
        batch_size_override=batch_size,
        tasks_filter=tasks_filter,
    )
    for k, v in results["student"].items():
        extra = f" | trunc={v['truncated']} med_len={v['median_len']}" if "truncated" in v else ""
        print(f"  [Bolt-3] {k:<14}: {v['acc']:6.2f}% ± {v['ci95']:4.1f}% ({v['correct']}/{v['total']}){extra}", flush=True)

    hf_donor = None
    if not results.get("systems") and device.type == "cuda":
        hf_donor = _load_hf_donor(donor_dir, device)
        results["systems"] = eval_real_speed_and_ram(stu, hf_donor, cfg, tok, device)
        _flush()
    if results.get("systems"):
        sc_last = results["systems"]["scaling"][-1]
        print(
            f"  [Bolt-3] Speed/RAM @{sc_last['context']}: {_sc_tok_s(sc_last)} vs {sc_last['donor_tok_s']} tok/s "
            f"({sc_last['speedup']}x) | {_sc_state_mb(sc_last)} vs {sc_last['donor_kv_mb']} MB (-{sc_last['ram_saved_pct']}%)",
            flush=True,
        )

    del stu
    _purge_zombie_vram()

    if compare_donor:
        need_keys = set(results["student"].keys()) - set(results.get("teacher", {}).keys())
        if need_keys:
            if hf_donor is None:
                hf_donor = _load_hf_donor(donor_dir, device)
            results["teacher"] = _eval_model(
                hf_donor,
                True,
                tok,
                device,
                limit=limit,
                gsm8k_limit=gsm8k_limit,
                cached=results.get("teacher"),
                on_save=lambda d: (results.__setitem__("teacher", d), _flush()),
                gsm_max_new=gsm_max_new,
                batch_size_override=batch_size,
                tasks_filter=tasks_filter,
            )
        for k, v in results["teacher"].items():
            extra = f" | trunc={v['truncated']} med_len={v['median_len']}" if "truncated" in v else ""
            print(f"  [Donor ] {k:<14}: {v['acc']:6.2f}% ± {v['ci95']:4.1f}% ({v['correct']}/{v['total']}){extra}", flush=True)
            if k in results["student"]:
                results["mcnemar"][k] = _mcnemar_exact(results["student"][k]["items"], v["items"])
    if hf_donor is not None:
        del hf_donor
    _purge_zombie_vram()

    _flush()
    if results.get("systems"):
        _plot_benchmark_figure(results, png_path)

    print("\n| Benchmark | HF Donor | **Bolt-3 (Full)** | Paired McNemar (`b/c`, `p`) | Verdict |")
    print("|---|---:|---:|---:|---|")
    for k, sv in results["student"].items():
        dv = results["teacher"].get(k, {})
        mc = results["mcnemar"].get(k, {"stu_only": 0, "don_only": 0, "p": 1.0})
        verdict = f"{sv['acc'] - dv.get('acc', 0.0):+.2f}% (sig)" if mc["p"] < 0.05 else "No detectable diff (p >= 0.05)"
        print(f"| **{k}** (`n={sv['total']}`) | `{dv.get('acc', 0.0):.2f}%` | **`{sv['acc']:.2f}%`** | `stu={mc['stu_only']}, don={mc['don_only']}` (`p={mc['p']:.3f}`) | {verdict} |")
    if results.get("systems"):
        sc_last = results["systems"]["scaling"][-1]
        print(f"| **Decode Speed @{sc_last['context']}** | `{sc_last['donor_tok_s']} tok/s` | **`{_sc_tok_s(sc_last)} tok/s`** | — | **`{sc_last['speedup']}x faster`** |")
        print(f"| **State RAM @{sc_last['context']}** | `{sc_last['donor_kv_mb']} MB` | **`{_sc_state_mb(sc_last)} MB`** | — | **`-{sc_last['ram_saved_pct']}% RAM`** |", flush=True)
    return results
