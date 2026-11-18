"""Checkpoint-compatible sensor/motion event backbone and preprocessing.

The frozen checkpoint computes both spatial and motion paths. The outer model
configuration determines which representation feeds classification.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import event_features as r2

SCOPE = 1024
CIN = 7
VMAX, N_BLOCKS = 8.0, 32
S_FINE, S_COARSE = 4, 16
W_IMG, H_IMG, T_TAU = 352.0, 260.0, 125.0

NX_FINE = math.ceil(W_IMG / S_FINE)          # 88
NY_FINE = math.ceil(H_IMG / S_FINE)          # 65
N_CF = NX_FINE * NY_FINE                      # 5720

NX_COARSE = math.ceil(W_IMG / S_COARSE)      # 22
NY_COARSE = math.ceil(H_IMG / S_COARSE)      # 17
N_CC = NX_COARSE * NY_COARSE                 # 374


def morton_key(qx, qy, qt, bits=10):
    k = torch.zeros_like(qx)
    for b in range(bits):
        k |= ((qx >> b) & 1) << (3 * b)
        k |= ((qy >> b) & 1) << (3 * b + 1)
        k |= ((qt >> b) & 1) << (3 * b + 2)
    return k


def layout_of(x, y, tau, S=4):
    qx = torch.clamp((x / S).floor().long(), 0, 1023)
    qy = torch.clamp((y / S).floor().long(), 0, 1023)
    qt = torch.clamp(tau.floor().long(), 0, 1023)
    key = morton_key(qx, qy, qt)
    order = torch.argsort(key)
    n = len(x)
    rows = math.ceil(n / SCOPE) * SCOPE
    row_all = torch.empty(n, dtype=torch.long, device=x.device)
    row_all[order] = torch.arange(n, device=x.device)
    return row_all, rows, n


def place(v, row, rows):
    o = torch.zeros((rows,) + v.shape[1:], dtype=v.dtype, device=v.device)
    o[row] = v
    return o


def vrows(row, rows):
    vr = torch.zeros(rows, dtype=torch.bool, device=row.device)
    vr[row] = True
    return vr


# ---------------------------------------------------------------- Modules
class NormalizedStem(nn.Module):
    def __init__(self, c_stem=64):
        super().__init__()
        self.w_c = nn.Linear(3, c_stem, bias=False)
        self.w_e = nn.Linear(CIN, c_stem, bias=False)
        self.norm = nn.LayerNorm(c_stem)
        self.act = nn.GELU()

    def forward(self, c, f):
        return self.act(self.norm(self.w_c(c) + self.w_e(f)))


class LocalSDPABlock(nn.Module):
    def __init__(self, channels=64, num_heads=4, scope=SCOPE, mlp_ratio=4.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scope = scope
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.gamma1 = nn.Parameter(torch.full((channels,), 0.1))
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(channels * mlp_ratio), channels)
        )
        self.gamma2 = nn.Parameter(torch.full((channels,), 0.1))
        self.post_norm = nn.LayerNorm(channels)

    def forward(self, x, valid_mask):
        device_type = x.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == 'cuda',
        ):
            N, C = x.shape
            num_scopes = N // self.scope
            x_norm = self.norm1(x)
            qkv = self.qkv(x_norm).view(num_scopes, self.scope, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.transpose(1, 2).reshape(N, C)
            x = x + self.gamma1 * self.proj(attn_out)
            x = x * valid_mask.unsqueeze(1)
            x = x + self.gamma2 * self.mlp(self.norm2(x))
            x = x * valid_mask.unsqueeze(1)
            return self.post_norm(x).float()


class EventwiseMLPBlock(nn.Module):
    """Per-event context control with no token-to-token communication."""

    def __init__(self, channels=64, mlp_ratio=4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.post_norm = nn.LayerNorm(channels)

    def forward(self, x, valid_mask):
        x = self.post_norm(x + self.mlp(self.norm(x)))
        return x * valid_mask.unsqueeze(1)


class VelocityHead(nn.Module):
    def __init__(self, c_stem=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c_stem, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, h):
        return VMAX * torch.tanh(self.mlp(h.float()))


class ConfidenceHead(nn.Module):
    def __init__(self, c_stem=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c_stem, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, h):
        return torch.sigmoid(self.mlp(h.float())).squeeze(1)


class DualFastEvidenceMemorySG(nn.Module):
    """Dual Spatial & Trajectory Memory with Trajectory-Verified Signed Evidence Accumulation.

    Modes:
      - mode='clean_base': zero evidence computation (standard DualFastMemorySG)
      - mode='pos_only'  : support evidence E^+ only (u^- = 0)
      - mode='signed'    : full signed evidence E^+ and E^- (contradiction evidence from cross-geometry disagreement)
    """
    def __init__(self, c_stem=64, lam=0.9, mode='signed', kappa=1.0, eps=1e-4):
        super().__init__()
        self.c_stem = c_stem
        self.lam = lam
        self.mode = mode
        self.kappa = kappa
        self.eps = eps

    def forward(self, h, hS, hT, cf_S, cc_S, cf_T, cc_T, blk, w, qm):
        dev = h.device
        N = h.shape[0]
        r_f_S = torch.zeros_like(h); r_c_S = torch.zeros_like(h)
        r_f_T = torch.zeros_like(h); r_c_T = torch.zeros_like(h)
        rho_out = torch.zeros((N, 1), dtype=torch.float32, device=dev)

        Mf_S = torch.zeros((N_CF, self.c_stem), dtype=torch.float32, device=dev)
        Mc_S = torch.zeros((N_CC, self.c_stem), dtype=torch.float32, device=dev)
        Wf_S = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)
        Wc_S = torch.zeros((N_CC, 1), dtype=torch.float32, device=dev)
        Ep_f_S = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)
        En_f_S = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)

        Mf_T = torch.zeros((N_CF, self.c_stem), dtype=torch.float32, device=dev)
        Mc_T = torch.zeros((N_CC, self.c_stem), dtype=torch.float32, device=dev)
        Wf_T = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)
        Wc_T = torch.zeros((N_CC, 1), dtype=torch.float32, device=dev)
        Ep_f_T = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)
        En_f_T = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)

        counts = torch.bincount(blk, minlength=N_BLOCKS)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(counts, 0)[:-1]])
        ends = starts + counts

        starts_cpu = starts.tolist()
        ends_cpu = ends.tolist()

        f_all = h.float()
        ww_all = w.unsqueeze(1)
        qm_all = qm.unsqueeze(1)

        compute_ev = (self.mode in ('pos_only', 'signed'))
        use_neg = (self.mode == 'signed')

        if compute_ev:
            norm_S = F.normalize(hS.detach().float(), dim=-1)
            norm_T = F.normalize(hT.detach().float(), dim=-1)
            cos_agree = (norm_S * norm_T).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            agree = 0.5 * (1.0 + cos_agree)
            agree_tilde = (1.0 - qm_all) + qm_all * agree
            u_pos_all = ww_all * agree_tilde
            u_neg_all = (ww_all * qm_all * (1.0 - agree)) if use_neg else torch.zeros_like(u_pos_all)
        else:
            u_pos_all = None; u_neg_all = None

        for st, en in zip(starts_cpu, ends_cpu):
            if st >= en:
                continue
            cf_b_S = cf_S[st:en]; cc_b_S = cc_S[st:en]
            cf_b_T = cf_T[st:en]; cc_b_T = cc_T[st:en]

            # Read-before-write memory features
            r_f_S[st:en] = Mf_S[cf_b_S] / Wf_S[cf_b_S].clamp_min(1e-6)
            r_c_S[st:en] = Mc_S[cc_b_S] / Wc_S[cc_b_S].clamp_min(1e-6)
            r_f_T[st:en] = Mf_T[cf_b_T] / Wf_T[cf_b_T].clamp_min(1e-6)
            r_c_T[st:en] = Mc_T[cc_b_T] / Wc_T[cc_b_T].clamp_min(1e-6)

            # Read signed verification evidence
            if compute_ev:
                ep_S = Ep_f_S[cf_b_S]; en_S = En_f_S[cf_b_S]
                ep_T = Ep_f_T[cf_b_T]; en_T = En_f_T[cf_b_T]
                rho_S = (ep_S - self.kappa * en_S) / (ep_S + self.kappa * en_S + self.eps)
                rho_T = (ep_T - self.kappa * en_T) / (ep_T + self.kappa * en_T + self.eps)
                qm_b = qm_all[st:en]
                rho_out[st:en] = (1.0 - qm_b) * rho_S + qm_b * rho_T

            f = f_all[st:en]; ww = ww_all[st:en]; f_ww = f * ww

            # Decay memory
            Mf_S = self.lam * Mf_S.detach(); Wf_S = self.lam * Wf_S.detach()
            Mc_S = self.lam * Mc_S.detach(); Wc_S = self.lam * Wc_S.detach()
            Mf_T = self.lam * Mf_T.detach(); Wf_T = self.lam * Wf_T.detach()
            Mc_T = self.lam * Mc_T.detach(); Wc_T = self.lam * Wc_T.detach()

            Mf_S.index_add_(0, cf_b_S, f_ww); Wf_S.index_add_(0, cf_b_S, ww)
            Mc_S.index_add_(0, cc_b_S, f_ww); Wc_S.index_add_(0, cc_b_S, ww)
            Mf_T.index_add_(0, cf_b_T, f_ww); Wf_T.index_add_(0, cf_b_T, ww)
            Mc_T.index_add_(0, cc_b_T, f_ww); Wc_T.index_add_(0, cc_b_T, ww)

            # Accumulate evidence
            if compute_ev and u_pos_all is not None and u_neg_all is not None:
                up = u_pos_all[st:en]; un = u_neg_all[st:en]
                Ep_f_S = self.lam * Ep_f_S.detach(); En_f_S = self.lam * En_f_S.detach()
                Ep_f_T = self.lam * Ep_f_T.detach(); En_f_T = self.lam * En_f_T.detach()
                Ep_f_S.index_add_(0, cf_b_S, up); En_f_S.index_add_(0, cf_b_S, un)
                Ep_f_T.index_add_(0, cf_b_T, up); En_f_T.index_add_(0, cf_b_T, un)

        return r_f_S, r_c_S, r_f_T, r_c_T, rho_out


class SensorEventBackbone(nn.Module):
    def __init__(self, c_stem=64, mode='signed', bound_mem=True, mem_scale_init=-1.1,
                 evidence_scale_init=-1.386, kappa=1.0,
                 encoder_context='local_attention'):
        super().__init__()
        if encoder_context not in {'local_attention', 'eventwise_mlp'}:
            raise ValueError(f'unknown encoder_context: {encoder_context}')
        self.c_stem = c_stem
        self.mode = mode
        self.bound_mem = bound_mem
        self.encoder_context = str(encoder_context)
        num_heads = max(2, c_stem // 16)
        self.stem = NormalizedStem(c_stem=c_stem)
        if self.encoder_context == 'eventwise_mlp':
            self.local_S = None
            self.local_T = None
            self.eventwise_S = EventwiseMLPBlock(channels=c_stem)
            self.eventwise_T = EventwiseMLPBlock(channels=c_stem)
        else:
            self.local_S = LocalSDPABlock(channels=c_stem, num_heads=num_heads, scope=SCOPE)
            self.local_T = LocalSDPABlock(channels=c_stem, num_heads=num_heads, scope=SCOPE)
            self.eventwise_S = None
            self.eventwise_T = None
        self.velocity = VelocityHead(c_stem=c_stem)
        self.motion_conf = ConfidenceHead(c_stem=c_stem)
        self.mem_scale_logit = nn.Parameter(torch.tensor(float(mem_scale_init)))

        # Evidence calibration: gamma_e = 0.5 * sigmoid(logit); init -1.386 -> gamma_e ~= 0.10
        self.evidence_scale_logit = nn.Parameter(torch.tensor(float(evidence_scale_init)))

        self.fuse = nn.Sequential(nn.Linear(3 * c_stem, 32), nn.GELU(), nn.Linear(32, 1))
        self.dual_mem = DualFastEvidenceMemorySG(c_stem=c_stem, mode=mode, kappa=kappa)
        self.mem_fuse = nn.Sequential(nn.Linear(4 * c_stem, 32), nn.GELU(), nn.Linear(32, 1))
        self.mem_proj = nn.Linear(2 * c_stem, c_stem, bias=False)
        self.cons = nn.Sequential(nn.Linear(3 * c_stem, 32), nn.GELU(), nn.Linear(32, 1))
        self.res = nn.Linear(c_stem, c_stem, bias=False)
        nn.init.zeros_(self.res.weight)
        self.pre = nn.Sequential(nn.Linear(2 * c_stem, 64), nn.GELU(), nn.Linear(64, 2))
        self.head = nn.Sequential(nn.Linear(2 * c_stem, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, b):
        h0_point = self.stem(b['coords_point'], b['feat_point'])

        # Spatial Pass
        h0_S = place(h0_point, b['row_S'], b['rows_S'])
        if self.encoder_context == 'eventwise_mlp':
            hS = self.eventwise_S(h0_S, b['vr_S'])[b['row_S']]
        else:
            hS = self.local_S(h0_S, b['vr_S'])[b['row_S']]

        # Motion Isolation
        v = self.velocity(hS.detach())
        qm = self.motion_conf(hS)

        # Detached Routing
        v_route = v.detach()
        xw = (b['x'] - v_route[:, 0] * b['dt']).clamp(0, W_IMG - 1)
        yw = (b['y'] - v_route[:, 1] * b['dt']).clamp(0, H_IMG - 1)
        row_T, rows_T, _ = layout_of(xw, yw, b['tau_raw'])
        coords_point_T = torch.stack([2 * xw / W_IMG - 1, 2 * yw / H_IMG - 1, b['tau_n']], 1)
        h0_point_T = self.stem(coords_point_T, b['feat_point'])
        h0_T = place(h0_point_T, row_T, rows_T)

        # Trajectory Pass
        if self.encoder_context == 'eventwise_mlp':
            hT = self.eventwise_T(h0_T, vrows(row_T, rows_T))[row_T]
        else:
            hT = self.local_T(h0_T, vrows(row_T, rows_T))[row_T]

        # Confidence Fallback Local Fusion
        alpha_gate = torch.sigmoid(self.fuse(torch.cat([hS, hT, (hS - hT).abs()], dim=1)))
        alpha = qm.unsqueeze(1) * alpha_gate
        h = (1 - alpha) * hS + alpha * hT

        # Pre-Head
        pre_feat = torch.cat([h0_point, h], 1)
        pre_logits = self.pre(pre_feat)
        with torch.no_grad():
            w = torch.softmax(pre_logits.detach(), -1)[:, 1]

        # Fused Dual Fast Memory with Signed Evidence Verification
        cf_S = ((b['x'] / S_FINE).long().clamp(0, NX_FINE - 1) * NY_FINE +
                (b['y'] / S_FINE).long().clamp(0, NY_FINE - 1))
        cc_S = ((b['x'] / S_COARSE).long().clamp(0, NX_COARSE - 1) * NY_COARSE +
                (b['y'] / S_COARSE).long().clamp(0, NY_COARSE - 1))
        cf_T = ((xw / S_FINE).long().clamp(0, NX_FINE - 1) * NY_FINE +
                (yw / S_FINE).long().clamp(0, NY_FINE - 1))
        cc_T = ((xw / S_COARSE).long().clamp(0, NX_COARSE - 1) * NY_COARSE +
                (yw / S_COARSE).long().clamp(0, NY_COARSE - 1))

        r_f_S, r_c_S, r_f_T, r_c_T, rho = self.dual_mem(h, hS, hT, cf_S, cc_S, cf_T, cc_T, b['blk'], w, qm)

        beta_gate = torch.sigmoid(self.mem_fuse(torch.cat([h, r_f_S, r_f_T, (r_f_S - r_f_T).abs()], dim=1)))
        beta = qm.unsqueeze(1) * beta_gate
        r_fused = (1 - beta) * r_f_S + beta * r_f_T
        r = r_fused + self.mem_proj(torch.cat([torch.max(r_f_S, r_c_S), torch.max(r_f_T, r_c_T)], 1)).float()

        # Bounded Memory Consistency Residual
        c = torch.sigmoid(self.cons(torch.cat([h, r, (h - r).abs()], 1)))
        mem_res_raw = c * self.res(r)
        gamma_mem = (0.25 * torch.sigmoid(self.mem_scale_logit)) if self.bound_mem else torch.tensor(1.0, device=h.device)
        mem_res = gamma_mem * mem_res_raw
        h_out = h + mem_res

        with torch.no_grad():
            res_ratio = (mem_res.norm(dim=1).mean() / h.norm(dim=1).mean().clamp_min(1e-6)).item()
            res_ratio_raw = (mem_res_raw.norm(dim=1).mean() / h.norm(dim=1).mean().clamp_min(1e-6)).item()

        # Final Head with Direct Pointwise Stem Skip
        final_feat = torch.cat([h0_point, h_out], 1)
        base_logits = self.head(final_feat)

        # Evidence Residual Logit Calibration (when mode != 'clean_base')
        if self.mode != 'clean_base':
            gamma_e = 0.5 * torch.sigmoid(self.evidence_scale_logit)
            rho_calib = gamma_e * torch.tanh(2.0 * rho)
            final_logits = torch.stack([
                base_logits[:, 0],
                base_logits[:, 1] + rho_calib.squeeze(1)
            ], dim=1)
        else:
            final_logits = base_logits
            gamma_e = torch.tensor(0.0, device=h.device)

        stats = {
            'res_ratio': float(res_ratio),
            'res_ratio_raw': float(res_ratio_raw),
            'gamma_mem': float(gamma_mem.detach().item()) if self.bound_mem else 1.0,
            'gamma_e': float(gamma_e.detach().item()),
            'rho_mean': float(rho.mean().item()),
            'rho_min': float(rho.min().item()),
            'rho_max': float(rho.max().item()),
            'w_mean': float(w.mean().item()),
            'qm_mean': float(qm.mean().item()),
            'w_res_norm': float(self.res.weight.norm().item())
        }
        return final_logits, pre_logits, v, qm, stats


def prep(clip, dev, mu, sd, tau_scale=64.0, n_blocks=N_BLOCKS, *, precomputed_features=None, clip_time_origin=None, clip_time_end=None, clip_time_center=None, clip_velocity=None, has_clip_velocity=None):
    locs, seg, fz = clip['locs'], clip['seg'], clip['feats']
    if int(n_blocks) != N_BLOCKS:
        raise ValueError(f"MoVeR requires n_blocks={N_BLOCKS} for the fixed memory layout")
    t = locs[:, 3].astype(np.float64)
    t0 = float(t.min()) if clip_time_origin is None else float(clip_time_origin)
    if np.any(t < t0):
        raise ValueError("clip_time_origin is later than a window event")
    tau = (t - t0) / float(tau_scale)
    if clip_time_end is None:
        tmax = max(1e-6, float(tau.max()))
    else:
        tmax = max(1e-6, (float(clip_time_end) - t0) / float(tau_scale))
    if float(tau.max()) > tmax + 1e-6:
        raise ValueError("clip_time_end is earlier than a window event")
    x = torch.as_tensor(locs[:, 1], dtype=torch.float32, device=dev)
    y = torch.as_tensor(locs[:, 2], dtype=torch.float32, device=dev)
    tau_t = torch.as_tensor(tau, dtype=torch.float32, device=dev)
    raw_features = (r2.stem_features(locs, fz, {'tau': tau, 't0': t0})
                    if precomputed_features is None else np.asarray(precomputed_features, dtype=np.float32))
    if raw_features.shape != (len(locs), CIN):
        raise ValueError(f"precomputed_features must have shape {(len(locs), CIN)}")
    F_feat = (raw_features - mu) / sd
    row_S, rows_S, n = layout_of(x, y, tau_t)
    coords_point = torch.stack([2 * x / W_IMG - 1, 2 * y / H_IMG - 1, tau_t / T_TAU], 1)
    feat_point = torch.as_tensor(F_feat, dtype=torch.float32, device=dev)
    blk = torch.clamp((tau_t / (tmax / N_BLOCKS)).floor().long(), 0, N_BLOCKS - 1)

    fg_mask = (seg == 1)
    if clip_velocity is None:
        v_gt = torch.zeros((n, 2), dtype=torch.float32, device=dev)
        has_target_vel = False
        if fg_mask.sum() >= 8:
            x_fg = locs[fg_mask, 1].astype(np.float64)
            y_fg = locs[fg_mask, 2].astype(np.float64)
            t_fg = tau[fg_mask]
            dt_fg = t_fg - t_fg.mean()
            den = float((dt_fg ** 2).sum())
            if den > 1e-4:
                vx_gt = float(((x_fg - x_fg.mean()) * dt_fg).sum() / den)
                vy_gt = float(((y_fg - y_fg.mean()) * dt_fg).sum() / den)
                v_gt[fg_mask, 0] = max(-VMAX, min(VMAX, vx_gt))
                v_gt[fg_mask, 1] = max(-VMAX, min(VMAX, vy_gt))
                has_target_vel = True
    else:
        velocity_array = np.asarray(clip_velocity, dtype=np.float32)
        if velocity_array.shape != (n, 2):
            raise ValueError(f"clip_velocity must have shape {(n, 2)}")
        v_gt = torch.as_tensor(velocity_array, dtype=torch.float32, device=dev)
        has_target_vel = bool(has_clip_velocity)

    time_center = float(tau_t.mean()) if clip_time_center is None else float(clip_time_center)
    return {'name': clip['name'], 'x': x, 'y': y, 'tau_raw': tau_t, 'tau_n': tau_t / T_TAU,
            'dt': tau_t - time_center, 'coords_point': coords_point, 'feat_point': feat_point,
            'row_S': row_S, 'rows_S': rows_S, 'vr_S': vrows(row_S, rows_S),
            'blk': blk, 'lab': torch.as_tensor(seg, dtype=torch.long, device=dev),
            'v_gt': v_gt, 'has_target_vel': has_target_vel, 'n': n}
