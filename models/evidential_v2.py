"""
Evidential Fusion V2 — Revised Architecture

Key fix: Per-modality evidence heads (scalar) → weighted fusion → single NIG head.

Why V1 failed:
  V1 had per-modality NIG heads predicting (γ_m, ν_m) independently,
  then fusing γ values by evidence. But individual modalities (especially
  audio/vision alone) are weak sentiment predictors → all ν_m are similarly
  low → gating reverts to near-uniform → no gain over concat.

V2 fix:
  Evidence e_m represents the "informativeness/quality" of modality m's
  representation h_m, NOT its independent prediction ability.
  A single NIG head after fusion handles the actual prediction.

  e_m = softplus(MLP_evidence(h_m))     ← "how useful is this representation?"
  h_fused = Σ (e_m/Σe_j) · h_m         ← evidence-weighted fusion
  (γ, ν, α, β) = NIG_head(h_fused)     ← single prediction with uncertainty

Reference: Deep Evidential Regression (Amini et al., NeurIPS 2020)
Author: Research Project A — Direction 1 (Evidential Gating V2)
Date: 2026-07-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CASP'))
from modules.transformer import TransformerEncoder


# ─── NIG Loss Functions (inlined from V1 to avoid import issues) ────

class NIGLoss(nn.Module):
    """Negative log-likelihood of Normal-Inverse-Gamma distribution."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y, gamma, nu, alpha, beta):
        nu = nu + self.eps
        alpha = alpha + self.eps
        beta = beta + self.eps
        omega = 2.0 * beta * (1.0 + nu)
        pi_term = 0.5 * (torch.log(torch.tensor(torch.pi, device=y.device)) - torch.log(nu))
        alpha_log_omega = alpha * torch.log(omega + self.eps)
        error_term = (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega + self.eps)
        lgamma_alpha = torch.lgamma(alpha + self.eps)
        lgamma_alpha_plus_half = torch.lgamma(alpha + 0.5 + self.eps)
        loss = pi_term - alpha_log_omega + error_term + lgamma_alpha - lgamma_alpha_plus_half
        return loss.mean()


class EvidenceRegularizer(nn.Module):
    """Evidence regularizer: penalizes high evidence on incorrect predictions."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y, gamma, nu, alpha):
        error = torch.abs(y - gamma)
        evidence_penalty = (2.0 * nu + alpha)
        return (error * evidence_penalty).mean()


# ─── Evidence Head ──────────────────────────────────────────────────

class EvidenceHead(nn.Module):
    """
    Maps a modality's hidden representation to a scalar evidence value.

    evidence = softplus(MLP(h))

    The evidence represents "how informative/trustworthy is this modality's
    representation for the fusion task?" — NOT "how well can this modality
    alone predict sentiment?"

    Using softplus ensures e > 0, with a small floor for numerical stability.
    """

    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 16)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h):
        """
        Args:
            h: [batch_size, input_dim]
        Returns:
            evidence: [batch_size, 1] — non-negative evidence scalar
        """
        raw = self.net(h)
        evidence = F.softplus(raw) + 1e-6  # ensure > 0
        return evidence


# ─── Revised Evidential Model ───────────────────────────────────────

class EvidentialFusionV2(nn.Module):
    """
    Revised evidential fusion: per-modality evidence → weighted fusion → single NIG.

    Architecture:
    1. Per-modality Conv1D projection + Transformer encoder (same as CASP)
    2. Per-modality EvidenceHead → e_m (scalar evidence)
    3. Evidence-weighted fusion: h_fused = Σ (e_m/Σe_j) · h_m
    4. Single NIGHead → (γ, ν, α, β)
    5. Final prediction: residual γ + MLP(h_fused)

    Modes:
      'evidential': full evidential fusion (NIG head + NIG loss)
      'evidential_l1': evidence-weighted fusion + L1 loss (ablation)
      'concat': CASP-style concat baseline (for comparison)
    """

    def __init__(
        self,
        orig_dim,
        output_dim=1,
        proj_dim=40,
        num_heads=8,
        layers=5,
        relu_dropout=0.1,
        embed_dropout=0.25,
        res_dropout=0.1,
        out_dropout=0.1,
        attn_dropout=0.1,
        mode='evidential',
        evidence_hidden=32,
        nig_hidden=32,
    ):
        super().__init__()

        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mods = len(orig_dim)
        self.mode = mode

        # ---- Projection Layers ----
        self.proj = nn.ModuleList([
            nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
            for i in range(self.num_mods)
        ])

        # ---- Per-modality Transformer Encoders ----
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim=proj_dim,
                num_heads=num_heads,
                layers=layers,
                attn_dropout=attn_dropout,
                res_dropout=res_dropout,
                relu_dropout=relu_dropout,
                embed_dropout=embed_dropout,
            )
            for _ in range(self.num_mods)
        ])

        # ---- Evidence Heads (NEW: per-modality scalar evidence) ----
        if 'evidential' in mode:
            self.evidence_heads = nn.ModuleList([
                EvidenceHead(input_dim=proj_dim, hidden_dim=evidence_hidden)
                for _ in range(self.num_mods)
            ])

        # ---- NIG Head (single, after fusion) ----
        if mode == 'evidential':
            self.nig_head = NIGHeadV2(
                input_dim=proj_dim,
                hidden_dim=nig_hidden
            )

        # ---- Output Head ----
        if mode in ['evidential', 'evidential_l1']:
            # Residual projection: MLP on fused representation + skip from gamma
            self.output_proj = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(out_dropout),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, output_dim),
            )
        elif mode == 'concat':
            self.output_proj = nn.Sequential(
                nn.Linear(self.num_mods * proj_dim, proj_dim * 2),
                nn.ReLU(),
                nn.Dropout(out_dropout),
                nn.Linear(proj_dim * 2, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, output_dim),
            )

    def encode_modality(self, x_i, encoder_idx):
        """Encode a single modality: projection → transformer → first-timestep pooling."""
        x_i = x_i.transpose(1, 2)
        x_i = self.proj[encoder_idx](x_i)
        x_i = x_i.permute(2, 0, 1)
        h_full = self.encoders[encoder_idx](x_i)
        h_pooled = h_full[0]
        return h_pooled, h_full

    def _encode_all(self, x):
        """
        Encode all modalities. Handle missing modalities.

        Returns:
            hs: [B, 3, proj_dim] all hidden states (zero for missing)
            evidence: [B, 3, 1] evidence values (zero for missing)
            mask: [B, 3] binary mask (1=present)
        """
        batch_size = x[0].shape[0]
        device = x[0].device
        hs = []
        evidence = []

        for i in range(self.num_mods):
            is_missing = (x[i].abs().sum() < 1e-8).item()

            if is_missing:
                hs.append(torch.zeros(batch_size, self.proj_dim, device=device))
                evidence.append(torch.zeros(batch_size, 1, device=device))
            else:
                h_pooled, _ = self.encode_modality(x[i], i)
                hs.append(h_pooled)
                if hasattr(self, 'evidence_heads'):
                    e = self.evidence_heads[i](h_pooled)
                else:
                    e = torch.ones(batch_size, 1, device=device)
                evidence.append(e)

        hs = torch.stack(hs, dim=1)         # [B, 3, proj_dim]
        evidence = torch.stack(evidence, dim=1)  # [B, 3, 1]
        mask = (evidence.squeeze(-1) > 1e-8).float()  # [B, 3]

        return hs, evidence, mask

    def _evidential_fuse(self, hs, evidence, mask):
        """
        Evidence-weighted fusion.

        w_m = e_m / Σ e_j    (if any evidence > 0, else uniform)
        h_fused = Σ w_m · h_m
        """
        total_evidence = evidence.sum(dim=1, keepdim=True)  # [B, 1, 1]

        # When all evidence is zero (all modalities missing), use uniform
        uniform = torch.ones_like(evidence) / self.num_mods
        weighted = evidence / (total_evidence + 1e-8)
        weights = torch.where(total_evidence > 1e-8, weighted, uniform)

        h_fused = (hs * weights).sum(dim=1)  # [B, proj_dim]
        weights_2d = weights.squeeze(-1)      # [B, 3]

        return h_fused, weights_2d

    def forward(self, x, return_evidence=False, return_nig=False):
        """
        Args:
            x: list of [text, audio, vision] tensors
        Returns:
            output: [B, 1] prediction
            (+ evidence weights, + NIG params)
        """
        hs, evidence, mask = self._encode_all(x)
        h_fused, weights = self._evidential_fuse(hs, evidence, mask)

        if self.mode == 'evidential':
            gamma, nu, alpha, beta = self.nig_head(h_fused)
            residual = self.output_proj(h_fused)
            output = gamma + residual

            results = (output,)
            if return_evidence:
                results += (weights, evidence.squeeze(-1))
            if return_nig:
                results += ((gamma, nu, alpha, beta),)
            return results[0] if len(results) == 1 else results

        elif self.mode == 'evidential_l1':
            output = self.output_proj(h_fused)
            if return_evidence:
                return output, weights, evidence.squeeze(-1)
            return output

        elif self.mode == 'concat':
            h_cat = hs.reshape(hs.shape[0], -1)
            output = self.output_proj(h_cat)
            return output


# ─── NIG Head V2 (single, after fusion) ────────────────────────────

class NIGHeadV2(nn.Module):
    """
    Single NIG head operating on the fused representation.

    Outputs (γ, ν, α, β) parameters of the Normal-Inverse-Gamma distribution.
    """

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gamma_head = nn.Linear(hidden_dim, 1)
        self.nu_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
            nn.Hardtanh(min_val=1.0 + 1e-6, max_val=100.0)
        )
        self.beta_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, h_fused):
        """h_fused: [B, proj_dim] → (γ, ν, α, β)"""
        shared = self.shared(h_fused)
        gamma = self.gamma_head(shared)
        nu = self.nu_head(shared) + 1e-6
        alpha = self.alpha_head(shared) + 1.0 + 1e-6
        beta = self.beta_head(shared) + 1e-6
        return gamma, nu, alpha, beta


# ─── Training Loss ──────────────────────────────────────────────────

class EvidentialLossV2(nn.Module):
    """
    Combined loss for V2 evidential training.

    L = L_nig + λ_reg * L_evidence_reg + λ_l1 * L_l1

    L_nig: NIG negative log-likelihood on fused prediction
    L_evidence_reg: |y - γ| · (2ν + α) — penalize high confidence on errors
    L_l1: optional L1 loss for stable training
    """

    def __init__(self, reg_weight=0.1, l1_weight=0.0):
        super().__init__()
        self.nig_loss = NIGLoss()
        self.evidence_reg = EvidenceRegularizer()
        self.reg_weight = reg_weight
        self.l1_weight = l1_weight
        self.l1 = nn.L1Loss()

    def forward(self, output, nig_params, labels):
        """
        Args:
            output: [B, 1] final prediction
            nig_params: (gamma, nu, alpha, beta)
            labels: [B, 1] ground truth
        """
        gamma, nu, alpha, beta = nig_params

        loss_nig = self.nig_loss(labels, gamma, nu, alpha, beta)
        loss_reg = self.evidence_reg(labels, gamma, nu, alpha)

        total = loss_nig + self.reg_weight * loss_reg

        if self.l1_weight > 0:
            total += self.l1_weight * self.l1(output, labels)

        return total, {
            'nig': loss_nig.item(),
            'reg': loss_reg.item(),
            'total': total.item(),
        }


# ─── Factory ────────────────────────────────────────────────────────

def build_evidential_v2(orig_dim, mode='evidential', **kwargs):
    return EvidentialFusionV2(orig_dim=orig_dim, mode=mode, **kwargs)


# ─── Sanity Check ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Evidential Fusion V2 Sanity Check ===\n")

    orig_dim = [768, 512, 256]
    batch_size = 4
    seq_len = 50

    text = torch.randn(batch_size, seq_len, orig_dim[0])
    audio = torch.randn(batch_size, seq_len, orig_dim[1])
    vision = torch.randn(batch_size, seq_len, orig_dim[2])

    for mode in ['evidential', 'evidential_l1', 'concat']:
        print(f"\n--- Mode: {mode} ---")
        model = EvidentialFusionV2(orig_dim=orig_dim, mode=mode)
        model.eval()

        x = [text, audio, vision]

        if mode == 'evidential':
            output, weights, evid, (gamma, nu, alpha, beta) = model(
                x, return_evidence=True, return_nig=True
            )
            print(f"  Output: {output.shape}, γ: {gamma.shape}")
            print(f"  ν: [{nu.min().item():.3f}, {nu.max().item():.3f}]")
            print(f"  Evidence: [{evid.min().item():.3f}, {evid.max().item():.3f}]")
            print(f"  Weights: {weights[0].detach().numpy()}")

            # Missing text
            x_m = [torch.zeros_like(text), audio, vision]
            _, w_m, e_m = model(x_m, return_evidence=True)
            print(f"  Missing text weights: {w_m[0].detach().numpy()}")
            print(f"  Missing text evidence: {e_m[0].detach().numpy()}")

        elif mode == 'evidential_l1':
            output, weights, evid = model(x, return_evidence=True)
            print(f"  Output: {output.shape}")
            print(f"  Evidence: [{evid.min().item():.3f}, {evid.max().item():.3f}]")
            print(f"  Weights: {weights[0].detach().numpy()}")

        elif mode == 'concat':
            output = model(x)
            print(f"  Output: {output.shape}")

        total = sum(p.numel() for p in model.parameters())
        print(f"  Params: {total:,}")

    # Test loss
    print("\n--- Loss Test ---")
    loss_fn = EvidentialLossV2(reg_weight=0.1, l1_weight=0.01)
    model = EvidentialFusionV2(orig_dim=orig_dim, mode='evidential')
    output, _, _, nig_params = model(x, return_evidence=True, return_nig=True)
    labels = torch.randn(batch_size, 1)
    loss, details = loss_fn(output, nig_params, labels)
    print(f"  NIG: {details['nig']:.4f}, Reg: {details['reg']:.4f}")
    print(f"  All finite: {all(torch.isfinite(torch.tensor(v)).item() for v in details.values())}")

    print("\n=== All sanity checks passed! ===")
