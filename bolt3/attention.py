"""Bolt-3 Dual-Field Attention: Attention Sinks + Local Window, Diversity Softmax Attractors,
and Gershgorin Eviction-to-Delta Residual Associative Memory (M_res).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import RMSNorm, RoPE, apply_rope_pos, repeat_kv


class StaticKV:
    """Optional dynamic KV buffer used only if stepping a Softmax teacher replica."""

    def __init__(self, B: int, HKV: int, Tmax: int, HD: int, dev, dt=torch.bfloat16):
        self.k = torch.empty(B, HKV, Tmax, HD, device=dev, dtype=dt)
        self.v = torch.empty_like(self.k)
        self.n = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        L = k_new.shape[2]
        if self.n + L > self.k.shape[2]:
            new_max = max(self.n + L + 1024, self.k.shape[2] * 2)
            nk = torch.empty(
                self.k.shape[0], self.k.shape[1], new_max, self.k.shape[3],
                device=self.k.device, dtype=self.k.dtype,
            )
            nv = torch.empty_like(nk)
            nk[:, :, : self.n] = self.k[:, :, : self.n]
            nv[:, :, : self.n] = self.v[:, :, : self.n]
            self.k, self.v = nk, nv
        self.k[:, :, self.n : self.n + L] = k_new
        self.v[:, :, self.n : self.n + L] = v_new
        self.n += L
        return self.k[:, :, : self.n], self.v[:, :, : self.n]


class Bolt3Attn(nn.Module):
    def __init__(self, cfg: dict, max_T: int = 2048, S: int | None = None, C: int | None = None):
        super().__init__()
        D, H, HKV = int(cfg["dim"]), int(cfg["num_heads"]), int(cfg["num_kv_heads"])
        HD = int(cfg.get("head_dim") or (D // H))
        qkv_bias = bool(cfg.get("qkv_bias", False))
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.S = int(S if S is not None else cfg.get("bolt3_S", 128))
        self.C = int(C if C is not None else cfg.get("bolt3_C", 128))
        self.n_sinks = int(cfg.get("bolt3_n_sinks", 4))
        self.sim_k = float(cfg.get("bolt3_sim_k", 0.94))
        self.sim_v = float(cfg.get("bolt3_sim_v", 0.88))
        self.gamma_u = float(cfg.get("bolt3_gamma_u", 0.985))

        self.q_proj = nn.Linear(D, H * HD, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)
        if cfg.get("qk_norm", False):
            self.q_norm = RMSNorm(HD, float(cfg["eps"]))
            self.k_norm = RMSNorm(HD, float(cfg["eps"]))
        else:
            self.q_norm, self.k_norm = None, None
        self.rope = RoPE(
            HD,
            float(cfg["rope_theta"]),
            cfg.get("rope_scaling"),
            float(cfg.get("partial_rotary_factor", 1.0)),
        )

        self.kappa = nn.Parameter(torch.ones(HKV))
        self.log_gamma = nn.Parameter(torch.full((HKV,), 3.5))
        self.beta_proj = nn.Linear(D, HKV, bias=True)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, -1.0)
        self.eta_proj = nn.Linear(D, H, bias=True)
        nn.init.zeros_(self.eta_proj.weight)
        nn.init.zeros_(self.eta_proj.bias)
        self.res_norm = RMSNorm(HD, float(cfg["eps"]))
        with torch.no_grad():
            self.res_norm.weight.fill_(0.05)

    def gamma_delta(self) -> torch.Tensor:
        return (0.999 * torch.sigmoid(self.log_gamma)).clamp(1e-3, 0.999).float()

    def head_qkv(self, x: torch.Tensor):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.H, self.HD)
        k = self.k_proj(x).view(b, t, self.HKV, self.HD)
        v = self.v_proj(x).view(b, t, self.HKV, self.HD)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def _consolidate_functional(
        self,
        ak,
        av,
        mass,
        util,
        pm,
        act,
        m_res,
        k_raw,
        k_rope,
        v_old,
        pos_old,
        beta_old,
        inv_freq,
        diag_eye,
        gam,
        n_filled: int = 0,
        tok_valid: torch.Tensor | None = None,
    ):
        b, hkv, l_tok, hd = k_raw.shape
        bh = b * hkv
        dt = k_raw.dtype
        kr_f = k_raw.reshape(bh, l_tok, hd)
        kp_f = k_rope.reshape(bh, l_tok, hd)
        v_f = v_old.reshape(bh, l_tok, hd)
        p_f = pos_old.expand(b, hkv, l_tok).reshape(bh, l_tok).float()
        b_f = beta_old.reshape(bh, l_tok)
        scale = hd ** -0.5

        if ak is None:
            dev = k_raw.device
            ak = torch.zeros(bh, self.S, hd, device=dev, dtype=dt)
            av = torch.zeros(bh, self.S, hd, device=dev, dtype=dt)
            mass = torch.ones(bh, self.S, device=dev, dtype=torch.float32)
            util = torch.zeros(bh, self.S, device=dev, dtype=torch.float32)
            pm = torch.zeros(bh, self.S, device=dev, dtype=torch.float32)
            act = torch.zeros(bh, self.S, device=dev, dtype=torch.bool)
            m_res = torch.zeros(bh, hd, hd, device=dev, dtype=torch.float32)
            n_filled = 0

        kap = self.kappa.float().repeat(b).view(bh, 1, 1)

        # Left-padded batch inference path: per-row valid token mask prevents pad tokens
        # from occupying attractors or decaying/writing into M_res.
        if tok_valid is not None:
            row_has_valid = tok_valid.any(dim=-1)
            util = torch.where(row_has_valid.unsqueeze(-1), util * self.gamma_u, util)
            has_any_act = act.any(dim=-1, keepdim=True)
            k_read = apply_rope_pos(
                ak.view(b, hkv, self.S, hd), pm.view(b, hkv, self.S), inv_freq
            ).view(bh, self.S, hd)
            with torch.no_grad():
                sc = torch.bmm(kp_f, k_read.transpose(-1, -2)).float() * scale + kap * torch.log(
                    mass.clamp_min(1.0)
                ).unsqueeze(1)
                sc.masked_fill_(~act.unsqueeze(1), -1e9)
                v_hat = torch.where(
                    has_any_act.unsqueeze(-1),
                    torch.bmm(F.softmax(sc, dim=-1).to(dt), av).float(),
                    0.0,
                )
                v_f32 = v_f.float()
                rel_err = (v_f32 - v_hat).norm(dim=-1) / (v_f32.norm(dim=-1) + 1e-6)
                ak_n = F.normalize(ak.float(), dim=-1)
                cos_k = torch.bmm(F.normalize(kr_f.float(), dim=-1), ak_n.transpose(-1, -2))
                cos_k.masked_fill_(~act.unsqueeze(1), -1.0)
                best_sim_k, best_slot = cos_k.max(dim=-1)
                matched_av = torch.gather(av.float(), 1, best_slot.unsqueeze(-1).expand(-1, -1, hd))
                sim_v = F.cosine_similarity(v_f32, matched_av, dim=-1)
                var_score = F.normalize(v_f32, dim=-1).var(dim=-1)
                score_j = torch.where(
                    has_any_act,
                    rel_err * (1.0 - best_sim_k.clamp(0.0, 1.0)),
                    var_score,
                )
                init_u = torch.where(has_any_act, rel_err, torch.full_like(rel_err, 0.15))
                merge_mask = tok_valid & has_any_act.squeeze(-1).unsqueeze(-1) & (best_sim_k >= self.sim_k) & (sim_v >= self.sim_v)

            m_oh = F.one_hot(best_slot, num_classes=self.S).float() * merge_mask.float().unsqueeze(-1)
            slot_cnt = m_oh.sum(dim=1)
            hit = slot_cnt > 0
            new_m = mass + slot_cnt
            ak = torch.where(
                hit.unsqueeze(-1),
                ((mass.unsqueeze(-1) * ak.float() + torch.bmm(m_oh.transpose(1, 2), kr_f.float())) / new_m.unsqueeze(-1)).to(dt),
                ak,
            )
            av = torch.where(
                hit.unsqueeze(-1),
                ((mass.unsqueeze(-1) * av.float() + torch.bmm(m_oh.transpose(1, 2), v_f.float())) / new_m.unsqueeze(-1)).to(dt),
                av,
            )
            pm = torch.where(
                hit,
                (mass * pm + torch.bmm(m_oh.transpose(1, 2), p_f.unsqueeze(-1)).squeeze(-1)) / new_m,
                pm,
            )
            mass = torch.where(hit, new_m, mass)

            ev_mask = tok_valid & (~merge_mask)
            with torch.no_grad():
                ak_n2 = F.normalize(ak.float(), dim=-1)
                pair_cos = torch.bmm(ak_n2, ak_n2.transpose(-1, -2))
                pair_cos.masked_fill_(diag_eye | (~act.unsqueeze(1)) | (~act.unsqueeze(2)), -1.0)
                keep_val = torch.where(
                    act,
                    util * (1.0 - pair_cos.max(dim=-1).values.clamp(0.0, 0.98)),
                    torch.full_like(util, -1e9),
                )
                n_scores = torch.where(ev_mask, score_j, torch.full_like(score_j, -1e9))
                k_cand = min(l_tok, self.S)
                top_vals, top_toks = torch.topk(n_scores, k=k_cand, dim=-1, largest=True)
                bot_vals, bot_slots = torch.topk(keep_val, k=k_cand, dim=-1, largest=False)
                win = (top_vals > bot_vals) & (top_vals > -1e8)
                ins_mask = torch.zeros_like(ev_mask).scatter(1, top_toks, win)
                delta_mask = (ev_mask & (~ins_mask)).float()

            k_g = torch.gather(kr_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd))
            v_g = torch.gather(v_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd))
            p_g = torch.gather(p_f, 1, top_toks)
            u_g = torch.gather(init_u, 1, top_toks)
            k_o = torch.gather(ak, 1, bot_slots.unsqueeze(-1).expand(-1, -1, hd))
            v_o = torch.gather(av, 1, bot_slots.unsqueeze(-1).expand(-1, -1, hd))
            p_o = torch.gather(pm, 1, bot_slots)
            m_o = torch.gather(mass, 1, bot_slots)
            u_o = torch.gather(util, 1, bot_slots)
            a_o = torch.gather(act, 1, bot_slots)
            w3 = win.unsqueeze(-1)
            ak = ak.scatter(1, bot_slots.unsqueeze(-1).expand(-1, -1, hd), torch.where(w3, k_g, k_o))
            av = av.scatter(1, bot_slots.unsqueeze(-1).expand(-1, -1, hd), torch.where(w3, v_g, v_o))
            pm = pm.scatter(1, bot_slots, torch.where(win, p_g, p_o))
            mass = mass.scatter(1, bot_slots, torch.where(win, torch.ones_like(m_o), m_o))
            util = util.scatter(1, bot_slots, torch.where(win, u_g, u_o))
            act = act.scatter(1, bot_slots, torch.where(win, torch.ones_like(a_o), a_o))

            k_ev = F.normalize(kr_f.float(), dim=-1)
            w_ev = b_f.float() * delta_mask
            gram = torch.bmm(k_ev, k_ev.transpose(-1, -2)).abs()
            gersh = torch.bmm(gram, w_ev.unsqueeze(-1)).squeeze(-1).clamp_min(1.0)
            beta_safe = (w_ev / gersh).unsqueeze(-1)
            v_pred = torch.bmm(k_ev, m_res)
            err_v = (v_f.float() - v_pred) * beta_safe
            gam_bh = gam.repeat(b).view(bh, 1, 1)
            gam_eff = torch.where(row_has_valid.view(bh, 1, 1), gam_bh, torch.ones_like(gam_bh))
            m_res = gam_eff * m_res + torch.bmm(k_ev.transpose(1, 2), err_v)
            return ak, av, mass, util, pm, act, m_res, int(act.sum(dim=-1).min().item())

        # Standard unpadded branchless path (used during training and unpadded decode)
        util = util * self.gamma_u
        eligible = torch.ones(bh, l_tok, device=k_raw.device, dtype=torch.bool)

        if n_filled < self.S:
            with torch.no_grad():
                n_empty = self.S - n_filled
                m = min(n_empty, l_tok)
                if n_filled > 0:
                    k_read = apply_rope_pos(
                        ak.view(b, hkv, self.S, hd), pm.view(b, hkv, self.S), inv_freq
                    ).view(bh, self.S, hd)
                    sc = torch.bmm(kp_f, k_read.transpose(-1, -2)).float() * scale + kap * torch.log(
                        mass.clamp_min(1.0)
                    ).unsqueeze(1)
                    sc.masked_fill_(~act.unsqueeze(1), -1e9)
                    v_hat = torch.bmm(F.softmax(sc, dim=-1).to(dt), av).float()
                    v_f32 = v_f.float()
                    rel_err = (v_f32 - v_hat).norm(dim=-1) / (v_f32.norm(dim=-1) + 1e-6)
                    cos_k = torch.bmm(
                        F.normalize(kr_f.float(), dim=-1),
                        F.normalize(ak.float(), dim=-1).transpose(-1, -2),
                    )
                    cos_k.masked_fill_(~act.unsqueeze(1), 0.0)
                    score_j = rel_err * (1.0 - cos_k.max(dim=-1).values.clamp(0.0, 1.0))
                    init_u = rel_err
                else:
                    score_j = F.normalize(v_f.float(), dim=-1).var(dim=-1)
                    init_u = torch.full((bh, l_tok), 0.15, device=k_raw.device, dtype=torch.float32)
                top_toks = torch.topk(score_j, k=m, dim=-1, largest=True).indices
                slot_ids = torch.arange(n_filled, n_filled + m, device=k_raw.device).unsqueeze(0).expand(bh, -1)

            ak = ak.scatter(
                1,
                slot_ids.unsqueeze(-1).expand(-1, -1, hd),
                torch.gather(kr_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd)),
            )
            av = av.scatter(
                1,
                slot_ids.unsqueeze(-1).expand(-1, -1, hd),
                torch.gather(v_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd)),
            )
            pm = pm.scatter(1, slot_ids, torch.gather(p_f, 1, top_toks))
            mass = mass.scatter(1, slot_ids, torch.ones(bh, m, device=k_raw.device, dtype=torch.float32))
            util = util.scatter(1, slot_ids, torch.gather(init_u, 1, top_toks))
            act = act.scatter(1, slot_ids, torch.ones(bh, m, device=k_raw.device, dtype=torch.bool))
            n_filled = n_filled + m
            if l_tok <= m:
                return ak, av, mass, util, pm, act, m_res, n_filled
            eligible = ~torch.zeros(bh, l_tok, device=k_raw.device, dtype=torch.bool).scatter(1, top_toks, True)

        k_read = apply_rope_pos(
            ak.view(b, hkv, self.S, hd), pm.view(b, hkv, self.S), inv_freq
        ).view(bh, self.S, hd)
        with torch.no_grad():
            sc = torch.bmm(kp_f, k_read.transpose(-1, -2)).float() * scale + kap * torch.log(
                mass.clamp_min(1.0)
            ).unsqueeze(1)
            v_hat = torch.bmm(F.softmax(sc, dim=-1).to(dt), av).float()
            v_f32 = v_f.float()
            rel_err = (v_f32 - v_hat).norm(dim=-1) / (v_f32.norm(dim=-1) + 1e-6)
            ak_n = F.normalize(ak.float(), dim=-1)
            cos_k = torch.bmm(F.normalize(kr_f.float(), dim=-1), ak_n.transpose(-1, -2))
            best_sim_k, best_slot = cos_k.max(dim=-1)
            matched_av = torch.gather(av.float(), 1, best_slot.unsqueeze(-1).expand(-1, -1, hd))
            sim_v = F.cosine_similarity(v_f32, matched_av, dim=-1)
            score_j = rel_err * (1.0 - best_sim_k.clamp(0.0, 1.0))
            merge_mask = eligible & (best_sim_k >= self.sim_k) & (sim_v >= self.sim_v)

        m_oh = F.one_hot(best_slot, num_classes=self.S).float() * merge_mask.float().unsqueeze(-1)
        slot_cnt = m_oh.sum(dim=1)
        k_sum = torch.bmm(m_oh.transpose(1, 2), kr_f.float())
        v_sum = torch.bmm(m_oh.transpose(1, 2), v_f.float())
        p_sum = torch.bmm(m_oh.transpose(1, 2), p_f.unsqueeze(-1)).squeeze(-1)
        hit = slot_cnt > 0
        new_m = mass + slot_cnt
        ak = torch.where(
            hit.unsqueeze(-1), ((mass.unsqueeze(-1) * ak.float() + k_sum) / new_m.unsqueeze(-1)).to(dt), ak
        )
        av = torch.where(
            hit.unsqueeze(-1), ((mass.unsqueeze(-1) * av.float() + v_sum) / new_m.unsqueeze(-1)).to(dt), av
        )
        pm = torch.where(hit, (mass * pm + p_sum) / new_m, pm)
        mass = torch.where(hit, new_m, mass)

        ev_mask = eligible & (~merge_mask)
        with torch.no_grad():
            ak_n2 = F.normalize(ak.float(), dim=-1)
            pair_cos = torch.bmm(ak_n2, ak_n2.transpose(-1, -2)).masked_fill(diag_eye, -1.0)
            keep_val = util * (1.0 - pair_cos.max(dim=-1).values.clamp(0.0, 0.98))
            n_scores = torch.where(ev_mask, score_j, torch.full_like(score_j, -1e9))
            k_cand = min(l_tok, self.S)
            top_vals, top_toks = torch.topk(n_scores, k=k_cand, dim=-1, largest=True)
            bot_vals, bot_slots = torch.topk(keep_val, k=k_cand, dim=-1, largest=False)
            win = (top_vals > bot_vals) & (top_vals > -1e8)
            ins_mask = torch.zeros_like(ev_mask).scatter(1, top_toks, win)
            delta_mask = (ev_mask & (~ins_mask)).float()

        k_g = torch.gather(kr_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd))
        v_g = torch.gather(v_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd))
        p_g = torch.gather(p_f, 1, top_toks)
        k_o = torch.gather(ak, 1, bot_slots.unsqueeze(-1).expand(-1, -1, hd))
        v_o = torch.gather(av, 1, bot_slots.unsqueeze(-1).expand(-1, -1, hd))
        p_o = torch.gather(pm, 1, bot_slots)
        m_o = torch.gather(mass, 1, bot_slots)
        u_o = torch.gather(util, 1, bot_slots)
        w3 = win.unsqueeze(-1)
        ak = ak.scatter(1, bot_slots.unsqueeze(-1).expand(-1, -1, hd), torch.where(w3, k_g, k_o))
        av = av.scatter(1, bot_slots.unsqueeze(-1).expand(-1, -1, hd), torch.where(w3, v_g, v_o))
        pm = pm.scatter(1, bot_slots, torch.where(win, p_g, p_o))
        mass = mass.scatter(1, bot_slots, torch.where(win, torch.ones_like(m_o), m_o))
        util = util.scatter(1, bot_slots, torch.where(win, torch.gather(rel_err, 1, top_toks), u_o))

        k_ev = F.normalize(kr_f.float(), dim=-1)
        w_ev = b_f.float() * delta_mask
        gram = torch.bmm(k_ev, k_ev.transpose(-1, -2)).abs()
        gersh = torch.bmm(gram, w_ev.unsqueeze(-1)).squeeze(-1).clamp_min(1.0)
        beta_safe = (w_ev / gersh).unsqueeze(-1)
        v_pred = torch.bmm(k_ev, m_res)
        err_v = (v_f.float() - v_pred) * beta_safe
        gam_bh = gam.repeat(b).view(bh, 1, 1)
        m_res = gam_bh * m_res + torch.bmm(k_ev.transpose(1, 2), err_v)
        return ak, av, mass, util, pm, act, m_res, n_filled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, T, _ = x.shape
        q_raw, k_raw, v_all_in = self.head_qkv(x)
        q_rope, k_rope = self.rope(q_raw), self.rope(k_raw)
        beta_all = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        eta_all = torch.tanh(self.eta_proj(x)).transpose(1, 2).unsqueeze(-1)
        gam = self.gamma_delta()
        inv_freq = self.rope.inv_freq.to(x.device)
        diag_eye = torch.eye(self.S, device=x.device, dtype=torch.bool).unsqueeze(0)
        scale = self.HD ** -0.5

        sink_k = sink_v = ak = av = mass = util = pm = act = m_res = None
        prev_kr = prev_kp = prev_v = prev_pos = prev_b = None
        n_filled = 0
        ys = []

        for s in range(0, T, self.C):
            e = min(T, s + self.C)
            tc = e - s
            qc_r, qc_p = q_raw[:, :, s:e], q_rope[:, :, s:e]
            kc_r, kc_p = k_raw[:, :, s:e], k_rope[:, :, s:e]
            vc, bc = v_all_in[:, :, s:e], beta_all[:, :, s:e]
            ec = eta_all[:, :, s:e]
            pos_c = torch.arange(s, e, device=x.device, dtype=torch.float32)[None, None, :]

            if prev_kp is not None:
                loc_k = torch.cat([prev_kp, kc_p], dim=2)
                loc_v = torch.cat([prev_v, vc], dim=2)
                prev_len = prev_kp.shape[2]
            else:
                loc_k, loc_v, prev_len = kc_p, vc, 0

            qg = qc_p.view(b, self.HKV, self.G, tc, self.HD)
            s_loc = torch.matmul(qg.float(), loc_k.float().unsqueeze(2).transpose(-1, -2)) * scale
            qi = torch.arange(tc, device=x.device)[:, None]
            kj = torch.arange(loc_k.shape[2], device=x.device)[None, :]
            s_loc = s_loc.masked_fill(~(kj <= (prev_len + qi)).view(1, 1, 1, tc, -1), float("-inf"))

            sp, vp = [], []
            if sink_k is not None:
                sp.append(torch.matmul(qg.float(), sink_k.float().unsqueeze(2).transpose(-1, -2)) * scale)
                vp.append(sink_v)
            if ak is not None:
                ak_eff = apply_rope_pos(
                    ak.view(b, self.HKV, self.S, self.HD), pm.view(b, self.HKV, self.S), inv_freq
                )
                s_att = torch.matmul(qg.float(), ak_eff.float().unsqueeze(2).transpose(-1, -2)) * scale
                kap = self.kappa.float().view(1, self.HKV, 1, 1, 1)
                m_bias = kap * torch.log(mass.view(b, self.HKV, 1, 1, self.S).clamp_min(1.0))
                act_m = act.view(b, self.HKV, 1, 1, self.S)
                s_att = (s_att + m_bias).masked_fill(~act_m, float("-inf"))
                sp.append(s_att)
                vp.append(av.view(b, self.HKV, self.S, self.HD))

            sp.append(s_loc)
            vp.append(loc_v)
            scores = torch.cat(sp, dim=-1) if len(sp) > 1 else sp[0]
            v_cat = torch.cat(vp, dim=2) if len(vp) > 1 else vp[0]
            probs = F.softmax(scores, dim=-1)
            if ak is not None:
                with torch.no_grad():
                    att_off = sink_k.shape[2] if sink_k is not None else 0
                    att_hit = (
                        0.25
                        * probs[..., att_off : att_off + self.S]
                        .mean(dim=(2, 3))
                        .reshape(b * self.HKV, self.S)
                    )
                    util = torch.maximum(util, att_hit)
            yc = torch.matmul(probs.to(v_cat.dtype), v_cat.unsqueeze(2)).reshape(b, self.H, tc, self.HD)

            if m_res is not None:
                q_rn = F.normalize(qc_r.view(b, self.HKV, self.G, tc, self.HD).float(), dim=-1)
                ym = torch.einsum("bhgtc,bhcd->bhgtd", q_rn, m_res.view(b, self.HKV, self.HD, self.HD))
                ym = ym * ((1.0 - gam).view(1, self.HKV, 1, 1, 1) / (self.HD ** 0.5))
                ym_n = self.res_norm(ym.reshape(b, self.H, tc, self.HD))
                yc = yc + (ec * ym_n).to(yc.dtype)
            elif e == T:
                yc = yc + 0.0 * (
                    ec.sum() + self.res_norm.weight.sum() + self.kappa.sum() + gam.sum() + bc.sum()
                )

            ys.append(yc)

            if tc == self.C and e < T:
                if prev_kp is not None:
                    p_kr, p_kp, p_v, p_pos, p_b = prev_kr, prev_kp, prev_v, prev_pos, prev_b
                    if sink_k is None:
                        ns = min(self.n_sinks, p_kp.shape[2])
                        sink_k, sink_v = p_kp[:, :, :ns], p_v[:, :, :ns]
                        p_kr, p_kp, p_v, p_pos, p_b = (
                            p_kr[:, :, ns:],
                            p_kp[:, :, ns:],
                            p_v[:, :, ns:],
                            p_pos[:, :, ns:],
                            p_b[:, :, ns:],
                        )
                    ak_d = ak.detach() if ak is not None else None
                    av_d = av.detach() if av is not None else None
                    pm_d = pm.detach() if pm is not None else None
                    mr_d = m_res.detach() if m_res is not None else None
                    ak, av, mass, util, pm, act, m_res, n_filled = self._consolidate_functional(
                        ak_d,
                        av_d,
                        mass,
                        util,
                        pm_d,
                        act,
                        mr_d,
                        p_kr,
                        p_kp,
                        p_v,
                        p_pos,
                        p_b,
                        inv_freq,
                        diag_eye,
                        gam,
                        n_filled=n_filled,
                    )
                prev_kr, prev_kp, prev_v, prev_pos, prev_b = kc_r, kc_p, vc, pos_c, bc

        y_out = torch.cat(ys, dim=2)
        return self.o_proj(y_out.transpose(1, 2).reshape(b, T, self.H * self.HD))


class Bolt3Decoder:
    """Constant-memory recurrent chunk/token decoder for Bolt-3 models."""

    def __init__(self, model, max_cache_pos: int = 65536, **kwargs):
        self.model = model.eval()
        self.max_T = int(model.config["max_T"])
        self.max_cache_pos = max_cache_pos
        self._diag_eye_cache: dict[torch.device, torch.Tensor] = {}
        self.C = int(model.config.get("bolt3_C", 128))

    @staticmethod
    def state_bytes(model) -> tuple[int, int]:
        flat, per_tok = 0, 0
        for blk in model.blocks:
            ly = blk.attn
            if isinstance(ly, Bolt3Attn):
                flat += (
                    ly.HKV * ly.HD * ly.HD * 4
                    + 2 * ly.S * ly.HKV * ly.HD * 2
                    + 3 * ly.S * ly.HKV * 4
                    + 6 * ly.C * ly.HKV * ly.HD * 2
                    + 2 * ly.n_sinks * ly.HKV * ly.HD * 2
                )
            else:
                per_tok += 2 * ly.HKV * ly.HD * 2
        return flat, per_tok

    @staticmethod
    def blank_state(
        model,
        batch_size: int,
        device,
        max_kv_len: int = 4096,
        state_dtype=torch.float32,
        **kwargs,
    ):
        states, kv_caches = [None] * len(model.blocks), [None] * len(model.blocks)
        dt = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        for li, blk in enumerate(model.blocks):
            ly = blk.attn
            if isinstance(ly, Bolt3Attn):
                states[li] = torch.zeros(batch_size, ly.HKV, ly.HD, ly.HD, device=device, dtype=state_dtype)
                kv_caches[li] = {
                    "ak": None,
                    "ak_eff": None,
                    "att_bias": None,
                    "av": None,
                    "mass": None,
                    "util": None,
                    "util_acc": None,
                    "pm": None,
                    "act": None,
                    "n_filled": 0,
                    "sink_k": None,
                    "sink_v": None,
                    "sink_valid": None,
                    "sink_filled": None,
                    "prev_kr": None,
                    "prev_kp": None,
                    "prev_v": None,
                    "prev_pos": None,
                    "prev_b": None,
                    "prev_valid": None,
                    "prev_has_pad": False,
                    "cur_kr": None,
                    "cur_kp": None,
                    "cur_v": None,
                    "cur_pos": None,
                    "cur_b": None,
                    "cur_valid": None,
                    "cur_has_pad": False,
                    "cur_len": 0,
                }
            else:
                kv_caches[li] = StaticKV(batch_size, ly.HKV, max_kv_len, ly.HD, device, dt=dt)
        return states, kv_caches

    @staticmethod
    def reset_state(states, kv_caches):
        for s in states:
            if s is not None:
                s.zero_()
        for c in kv_caches:
            if isinstance(c, dict):
                for k in list(c.keys()):
                    if k in ("n_filled", "cur_len"):
                        c[k] = 0
                    elif k in ("prev_has_pad", "cur_has_pad"):
                        c[k] = False
                    else:
                        c[k] = None
            elif isinstance(c, StaticKV):
                c.n = 0

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        history_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = logits.float()
        if repetition_penalty != 1.0 and history_ids is not None and history_ids.numel() > 0:
            for r in range(scores.shape[0]):
                seen = torch.unique(history_ids[r])
                sel = scores[r, seen]
                scores[r, seen] = torch.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)
        if temperature <= 1e-5:
            return scores.argmax(-1, keepdim=True)
        probs = F.softmax(scores / temperature, dim=-1)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cp = torch.cumsum(sp, dim=-1)
            sp[cp - sp > top_p] = 0.0
            sp.div_(sp.sum(dim=-1, keepdim=True).clamp_min(1e-12))
            return si.gather(-1, torch.multinomial(sp, 1))
        tv, ti = torch.topk(scores / temperature, min(top_k, scores.size(-1)), dim=-1)
        return ti.gather(-1, torch.multinomial(F.softmax(tv, -1), 1))

    @staticmethod
    def _refresh_attractor_cache(sc: dict, ly: Bolt3Attn, b: int, inv_freq: torch.Tensor):
        sc["ak_eff"] = apply_rope_pos(
            sc["ak"].view(b, ly.HKV, ly.S, ly.HD), sc["pm"].view(b, ly.HKV, ly.S), inv_freq
        )
        kap = ly.kappa.float().view(1, ly.HKV, 1, 1)
        mb = kap * torch.log(sc["mass"].view(b, ly.HKV, 1, ly.S).clamp_min(1.0))
        am = sc["act"].view(b, ly.HKV, 1, ly.S)
        sc["att_bias"] = mb.masked_fill(~am, float("-inf"))

    def step_block(
        self,
        li: int,
        blk,
        x: torch.Tensor,
        pos_start: int,
        states,
        kv_caches,
        pos_ids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Step one Transformer block on a chunk-aligned slice `(B, t_s, D)` where `cur_len + t_s <= C`."""
        b, t_s, _ = x.shape
        device = x.device
        h = blk.norm1(x)
        ly = blk.attn
        if isinstance(ly, Bolt3Attn):
            sc: dict = kv_caches[li]
            inv_freq = ly.rope.inv_freq.to(device)
            q_r, k_r, v_n = ly.head_qkv(h)
            if pos_ids is None:
                q_p, k_p = ly.rope(q_r, pos_start=pos_start), ly.rope(k_r, pos_start=pos_start)
                pos_n = (
                    torch.arange(pos_start, pos_start + t_s, device=device, dtype=torch.float32)
                    .view(1, 1, t_s)
                    .expand(b, 1, t_s)
                )
            else:
                pos_n = pos_ids.unsqueeze(1).float()
                q_p = apply_rope_pos(q_r, pos_n, inv_freq)
                k_p = apply_rope_pos(k_r, pos_n, inv_freq)

            beta_n = torch.sigmoid(ly.beta_proj(h)).transpose(1, 2)
            if valid_mask is not None:
                beta_n = beta_n * valid_mask.unsqueeze(1).to(beta_n.dtype)
            eta_n = torch.tanh(ly.eta_proj(h)).transpose(1, 2).unsqueeze(-1)
            gam = ly.gamma_delta()
            scale = ly.HD ** -0.5

            if sc["cur_kr"] is None:
                sc["cur_kr"] = torch.empty(b, ly.HKV, ly.C, ly.HD, device=device, dtype=k_r.dtype)
                sc["cur_kp"] = torch.empty_like(sc["cur_kr"])
                sc["cur_v"] = torch.empty_like(sc["cur_kr"])
                sc["cur_pos"] = torch.zeros(b, 1, ly.C, device=device, dtype=torch.float32)
                sc["cur_b"] = torch.empty(b, ly.HKV, ly.C, device=device, dtype=beta_n.dtype)
                sc["cur_valid"] = torch.ones(b, 1, ly.C, device=device, dtype=torch.bool)
                sc["cur_len"] = 0

            len_before = sc["cur_len"]
            new_len = len_before + t_s
            sc["cur_kr"][:, :, len_before:new_len] = k_r
            sc["cur_kp"][:, :, len_before:new_len] = k_p
            sc["cur_v"][:, :, len_before:new_len] = v_n
            sc["cur_pos"][:, :, len_before:new_len] = pos_n
            sc["cur_b"][:, :, len_before:new_len] = beta_n
            if valid_mask is not None:
                sc["cur_valid"][:, :, len_before:new_len] = valid_mask.unsqueeze(1)
                if not bool(valid_mask.all()):
                    sc["cur_has_pad"] = True
            else:
                sc["cur_valid"][:, :, len_before:new_len] = True
            sc["cur_len"] = new_len

            cur_kp_v = sc["cur_kp"][:, :, :new_len]
            cur_v_v = sc["cur_v"][:, :, :new_len]
            cur_val_v = sc["cur_valid"][:, :, :new_len]
            if sc["prev_kp"] is not None:
                loc_k = torch.cat([sc["prev_kp"], cur_kp_v], dim=2)
                loc_v = torch.cat([sc["prev_v"], cur_v_v], dim=2)
                loc_valid = torch.cat([sc["prev_valid"], cur_val_v], dim=2)
                prev_len = sc["prev_kp"].shape[2]
            else:
                loc_k, loc_v, loc_valid, prev_len = cur_kp_v, cur_v_v, cur_val_v, 0

            qg = q_p.view(b, ly.HKV, ly.G, t_s, ly.HD)
            s_loc = torch.matmul(qg.float(), loc_k.float().unsqueeze(2).transpose(-1, -2)) * scale
            has_pad = sc.get("prev_has_pad", False) or sc.get("cur_has_pad", False)
            if t_s > 1 or has_pad:
                qi = torch.arange(t_s, device=device)[:, None]
                kj = torch.arange(loc_k.shape[2], device=device)[None, :]
                cm = (kj <= (prev_len + len_before + qi)).view(1, 1, 1, t_s, -1)
                if has_pad:
                    q_ok = valid_mask if valid_mask is not None else torch.ones(b, t_s, device=device, dtype=torch.bool)
                    self_pos = (kj == (prev_len + len_before + qi)).view(1, 1, 1, t_s, -1)
                    cm = (cm & loc_valid.view(b, 1, 1, 1, -1)) | (self_pos & (~q_ok).view(b, 1, 1, t_s, 1))
                s_loc = s_loc.masked_fill(~cm, float("-inf"))

            sp, vp = [], []
            if sc["sink_k"] is not None:
                s_sink = torch.matmul(qg.float(), sc["sink_k"].float().unsqueeze(2).transpose(-1, -2)) * scale
                if sc.get("sink_valid") is not None and not bool(sc["sink_valid"].all()):
                    s_sink = s_sink.masked_fill(~sc["sink_valid"].view(b, 1, 1, 1, -1), float("-inf"))
                sp.append(s_sink)
                vp.append(sc["sink_v"])
            if sc["ak"] is not None:
                if sc.get("ak_eff") is None or sc.get("att_bias") is None:
                    self._refresh_attractor_cache(sc, ly, b, inv_freq)
                s_att = (
                    torch.matmul(qg.float(), sc["ak_eff"].float().unsqueeze(2).transpose(-1, -2)) * scale
                    + sc["att_bias"].view(b, ly.HKV, 1, 1, ly.S)
                )
                sp.append(s_att)
                vp.append(sc["av"].view(b, ly.HKV, ly.S, ly.HD))

            sp.append(s_loc)
            vp.append(loc_v)
            scores = torch.cat(sp, dim=-1) if len(sp) > 1 else sp[0]
            v_cat = torch.cat(vp, dim=2) if len(vp) > 1 else vp[0]
            probs = F.softmax(scores, dim=-1)

            if sc["ak"] is not None:
                att_off = sc["sink_k"].shape[2] if sc["sink_k"] is not None else 0
                att_p = probs[..., att_off : att_off + ly.S]
                if valid_mask is not None:
                    att_p = att_p * valid_mask.view(b, 1, 1, t_s, 1).float()
                att_sum = att_p.mean(dim=2).sum(dim=2).reshape(b * ly.HKV, ly.S)
                if sc.get("util_acc") is None:
                    sc["util_acc"] = att_sum
                else:
                    sc["util_acc"] = sc["util_acc"] + att_sum

            y = torch.matmul(probs.to(v_cat.dtype), v_cat.unsqueeze(2)).reshape(b, ly.H, t_s, ly.HD)

            if states[li] is not None:
                q_rn = F.normalize(q_r.view(b, ly.HKV, ly.G, t_s, ly.HD).float(), dim=-1)
                ym = torch.einsum("bhgtc,bhcd->bhgtd", q_rn, states[li]) * (
                    (1.0 - gam).view(1, ly.HKV, 1, 1, 1) / (ly.HD ** 0.5)
                )
                ym_n = ly.res_norm(ym.reshape(b, ly.H, t_s, ly.HD))
                y = y + (eta_n * ym_n).to(y.dtype)

            x = x + ly.o_proj(y.transpose(1, 2).reshape(b, t_s, ly.H * ly.HD))

            if new_len >= ly.C:
                if sc.get("util_acc") is not None and sc["util"] is not None:
                    sc["util"] = torch.maximum(sc["util"], 0.25 * (sc["util_acc"] / float(ly.C)))
                    sc["util_acc"].zero_()
                if sc["prev_kp"] is not None:
                    p_kr, p_kp, p_v, p_pos, p_b, p_val = (
                        sc["prev_kr"],
                        sc["prev_kp"],
                        sc["prev_v"],
                        sc["prev_pos"],
                        sc["prev_b"],
                        sc["prev_valid"],
                    )
                    all_valid_prev = bool(p_val.all())
                    if sc["sink_k"] is None and all_valid_prev:
                        ns = min(ly.n_sinks, p_kp.shape[2])
                        sc["sink_k"] = p_kp[:, :, :ns].clone()
                        sc["sink_v"] = p_v[:, :, :ns].clone()
                        sc["sink_valid"] = torch.ones(b, 1, ns, device=device, dtype=torch.bool)
                        sc["sink_filled"] = torch.full((b,), ns, device=device, dtype=torch.long)
                        p_kr, p_kp, p_v, p_pos, p_b = (
                            p_kr[:, :, ns:],
                            p_kp[:, :, ns:],
                            p_v[:, :, ns:],
                            p_pos[:, :, ns:],
                            p_b[:, :, ns:],
                        )
                        tok_valid_in = None
                    elif not all_valid_prev or (sc["sink_filled"] is not None and bool((sc["sink_filled"] < ly.n_sinks).any())):
                        if sc["sink_k"] is None:
                            ns = min(ly.n_sinks, p_kp.shape[2])
                            sc["sink_k"] = torch.zeros(b, ly.HKV, ns, ly.HD, device=device, dtype=p_kp.dtype)
                            sc["sink_v"] = torch.zeros_like(sc["sink_k"])
                            sc["sink_valid"] = torch.zeros(b, 1, ns, device=device, dtype=torch.bool)
                            sc["sink_filled"] = torch.zeros(b, device=device, dtype=torch.long)
                        p_val_work = p_val.clone()
                        ns = sc["sink_k"].shape[2]
                        for r in range(b):
                            sf = int(sc["sink_filled"][r].item())
                            if sf < ns:
                                v_idx = torch.nonzero(p_val_work[r, 0], as_tuple=False).view(-1)
                                take = min(ns - sf, int(v_idx.numel()))
                                if take > 0:
                                    sel = v_idx[:take]
                                    sc["sink_k"][r, :, sf : sf + take] = p_kp[r, :, sel]
                                    sc["sink_v"][r, :, sf : sf + take] = p_v[r, :, sel]
                                    sc["sink_valid"][r, 0, sf : sf + take] = True
                                    sc["sink_filled"][r] = sf + take
                                    p_val_work[r, 0, sel] = False
                        tok_valid_in = p_val_work.expand(b, ly.HKV, -1).reshape(b * ly.HKV, -1)
                    else:
                        tok_valid_in = None

                    eye = self._diag_eye_cache.get(device)
                    if eye is None:
                        eye = torch.eye(ly.S, device=device, dtype=torch.bool).unsqueeze(0)
                        self._diag_eye_cache[device] = eye
                    m_in = states[li].view(b * ly.HKV, ly.HD, ly.HD)
                    sc["ak"], sc["av"], sc["mass"], sc["util"], sc["pm"], sc["act"], m_out, sc["n_filled"] = (
                        ly._consolidate_functional(
                            sc["ak"],
                            sc["av"],
                            sc["mass"],
                            sc["util"],
                            sc["pm"],
                            sc["act"],
                            m_in,
                            p_kr,
                            p_kp,
                            p_v,
                            p_pos,
                            p_b,
                            inv_freq,
                            eye,
                            gam,
                            n_filled=sc.get("n_filled", 0),
                            tok_valid=tok_valid_in,
                        )
                    )
                    states[li] = m_out.view(b, ly.HKV, ly.HD, ly.HD)
                    self._refresh_attractor_cache(sc, ly, b, inv_freq)

                # Swap prev and cur preallocated buffers (zero tensor allocation)
                sc["prev_kr"], sc["cur_kr"] = sc["cur_kr"], sc["prev_kr"]
                sc["prev_kp"], sc["cur_kp"] = sc["cur_kp"], sc["prev_kp"]
                sc["prev_v"], sc["cur_v"] = sc["cur_v"], sc["prev_v"]
                sc["prev_pos"], sc["cur_pos"] = sc["cur_pos"], sc["prev_pos"]
                sc["prev_b"], sc["cur_b"] = sc["cur_b"], sc["prev_b"]
                sc["prev_valid"], sc["cur_valid"] = sc["cur_valid"], sc["prev_valid"]
                sc["prev_has_pad"], sc["cur_has_pad"] = sc.get("cur_has_pad", False), False
                sc["cur_len"] = 0
        else:
            q, k, v = ly.head_qkv(h)
            q, k = ly.rope(q, pos_start=pos_start), ly.rope(k, pos_start=pos_start)
            n_prev = kv_caches[li].n
            pk, pv = kv_caches[li].append(k, v)
            if t_s > 1:
                qi = torch.arange(t_s, device=device)[:, None]
                kj = torch.arange(n_prev + t_s, device=device)[None, :]
                cmask = (kj <= (n_prev + qi)).view(1, 1, t_s, n_prev + t_s)
                try:
                    y = F.scaled_dot_product_attention(
                        q, pk.to(q.dtype), pv.to(q.dtype), attn_mask=cmask, enable_gqa=(ly.G > 1)
                    )
                except TypeError:
                    y = F.scaled_dot_product_attention(
                        q, repeat_kv(pk.to(q.dtype), ly.G), repeat_kv(pv.to(q.dtype), ly.G), attn_mask=cmask
                    )
            else:
                try:
                    y = F.scaled_dot_product_attention(q, pk.to(q.dtype), pv.to(q.dtype), enable_gqa=(ly.G > 1))
                except TypeError:
                    y = F.scaled_dot_product_attention(
                        q, repeat_kv(pk.to(q.dtype), ly.G), repeat_kv(pv.to(q.dtype), ly.G)
                    )
            x = x + ly.o_proj(y.transpose(1, 2).reshape(b, t_s, ly.H * ly.HD))

        return x + blk.mlp(blk.norm2(x))

    @torch.no_grad()
    def step(
        self,
        tokens: torch.Tensor,
        pos_start: int,
        states,
        kv_caches,
        pos_ids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ):
        m = self.model
        t_tot = tokens.shape[1]
        if t_tot == 1:
            with torch.autocast("cuda", torch.bfloat16, enabled=tokens.is_cuda):
                x = m.embed_tokens(tokens)
                for li, blk in enumerate(m.blocks):
                    x = self.step_block(
                        li, blk, x, pos_start, states, kv_caches, pos_ids=pos_ids, valid_mask=valid_mask
                    )
                return m.norm(x), states, kv_caches

        outs = []
        s = 0
        pos = pos_start
        while s < t_tot:
            cur_len = 0
            for c in kv_caches:
                if isinstance(c, dict):
                    cur_len = c.get("cur_len", 0)
                    break
            take = min(t_tot - s, self.C - cur_len)
            p_slice = pos_ids[:, s : s + take] if pos_ids is not None else None
            v_slice = valid_mask[:, s : s + take] if valid_mask is not None else None
            with torch.autocast("cuda", torch.bfloat16, enabled=tokens.is_cuda):
                x = m.embed_tokens(tokens[:, s : s + take])
                for li, blk in enumerate(m.blocks):
                    x = self.step_block(
                        li, blk, x, pos, states, kv_caches, pos_ids=p_slice, valid_mask=v_slice
                    )
                outs.append(m.norm(x))
            s += take
            pos += take
        out_h = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return out_h, states, kv_caches

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, max_kv_len: int = 4096, attn_mask: torch.Tensor | None = None):
        b, t = ids.shape
        states, kv = self.blank_state(self.model, b, ids.device, max_kv_len=max(max_kv_len, t + 1024))
        if attn_mask is not None:
            valid_m = attn_mask.bool()
            pos_ids = (valid_m.long().cumsum(dim=-1) - 1).clamp_min(0)
        else:
            valid_m, pos_ids = None, None
        h, states, kv = self.step(ids, 0, states, kv, pos_ids=pos_ids, valid_mask=valid_m)
        return h, states, kv

    @torch.no_grad()
    def generate_batch(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.6,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_ids: set[int] | None = None,
        close_think_ids: list[int] | None = None,
        repetition_penalty: float = 1.0,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = ids.shape
        h, states, kv = self.prefill(ids, max_kv_len=T + max_new_tokens + 64, attn_mask=attn_mask)
        logits = self.model.lm_head(h[:, -1, :])
        eos = torch.tensor(sorted(eos_ids or ()), device=ids.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        lengths = torch.full((B,), max_new_tokens, dtype=torch.long, device=ids.device)
        end_think_id = close_think_ids[1] if (close_think_ids and len(close_think_ids) > 1) else None
        closed_think = torch.zeros(B, dtype=torch.bool, device=ids.device)
        q_pos = torch.full((B,), -1, dtype=torch.long, device=ids.device)
        budget_trigger = max(64, max_new_tokens - 64)
        hist = ids if repetition_penalty != 1.0 else None

        if attn_mask is not None:
            row_pos = attn_mask.long().sum(dim=-1, keepdim=True)
        else:
            row_pos = None

        out, pos = [], T
        for i in range(max_new_tokens):
            nxt = self._sample(
                logits,
                temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                history_ids=hist,
            )
            if close_think_ids:
                if i == budget_trigger:
                    need = (~closed_think) & (~done) & (q_pos < 0)
                    q_pos[need] = 0
                active_q = q_pos >= 0
                if bool(active_q.any()):
                    for r in torch.nonzero(active_q, as_tuple=False).view(-1).tolist():
                        nxt[r, 0] = close_think_ids[int(q_pos[r])]
                        q_pos[r] += 1
                        if q_pos[r] >= len(close_think_ids):
                            q_pos[r] = -1
                            closed_think[r] = True
                if end_think_id is not None:
                    closed_think |= nxt[:, 0] == end_think_id
            out.append(nxt)
            if hist is not None:
                hist = torch.cat([hist, nxt], dim=1)
            if eos.numel():
                hit = torch.isin(nxt[:, 0], eos) & ~done
                lengths[hit] = i + 1
                done |= hit
                if i % 8 == 7 and bool(done.all()):
                    break
            h, states, kv = self.step(nxt, pos, states, kv, pos_ids=row_pos)
            logits = self.model.lm_head(h[:, -1, :])
            pos += 1
            if row_pos is not None:
                row_pos = row_pos + 1
        return torch.cat(out, dim=1), lengths

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_ids: set[int] | None = None,
        close_think_ids: list[int] | None = None,
        repetition_penalty: float = 1.0,
    ):
        toks, n = self.generate_batch(
            prompt,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_ids=eos_ids,
            close_think_ids=close_think_ids,
            repetition_penalty=repetition_penalty,
        )
        return torch.cat([prompt, toks[:, : int(n[0].item())]], dim=1)

    @torch.no_grad()
    def continue_generate(
        self,
        ids: torch.Tensor,
        states,
        kv_caches,
        start_pos: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_ids: set[int] | None = None,
    ):
        h, states, kv_caches = self.step(ids, start_pos, states, kv_caches)
        pos = start_pos + ids.shape[1]
        logits = self.model.lm_head(h[:, -1, :])
        out = []
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k=top_k, top_p=top_p)
            out.append(nxt)
            if eos_ids and int(nxt[0, 0].item()) in eos_ids:
                break
            h, states, kv_caches = self.step(nxt, pos, states, kv_caches)
            logits = self.model.lm_head(h[:, -1, :])
            pos += 1
        toks = (
            torch.cat(out, dim=1)
            if out
            else torch.empty(ids.shape[0], 0, dtype=torch.long, device=ids.device)
        )
        return toks, states, kv_caches, pos

    @torch.no_grad()
    def memorize(self, ids: torch.Tensor, states, kv_caches, start_pos: int):
        pre_s = [s.clone() if s is not None else None for s in states]
        _, states, kv_caches = self.step(ids, start_pos, states, kv_caches)
        n_tok = ids.shape[1]
        pos = start_pos + n_tok
        traces = []
        for li, blk in enumerate(self.model.blocks):
            if isinstance(blk.attn, Bolt3Attn):
                gam = blk.attn.gamma_delta().view(1, -1, 1, 1)
                start_chunks = start_pos // blk.attn.C
                end_chunks = pos // blk.attn.C
                n_res_steps = max(0, max(0, end_chunks - 2) - max(0, start_chunks - 2))
                delta = states[li] - (gam ** n_res_steps) * pre_s[li]
                traces.append((pos, delta))
            else:
                traces.append(None)
        return states, kv_caches, pos, traces

    @torch.no_grad()
    def forget(self, traces, states, next_pos: int, kv_caches=None, pre_kv_lens=None):
        for li, tr in enumerate(traces):
            if tr is not None and states[li] is not None:
                end_p, delta = tr
                ly = self.model.blocks[li].attn
                gam = ly.gamma_delta().view(1, -1, 1, 1)
                end_chunks = end_p // ly.C
                next_chunks = next_pos // ly.C
                n_res_steps = max(0, max(0, next_chunks - 2) - max(0, end_chunks - 2))
                states[li] = states[li] - (gam ** n_res_steps) * delta
        return states
