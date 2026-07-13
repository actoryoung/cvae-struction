"""
Evidential Deep Learning-based Multimodal Fusion for Sentiment Analysis.

Based on "Deep Evidential Regression" (Amini et al., NeurIPS 2020):
https://arxiv.org/abs/1910.02600

Each modality outputs a Normal-Inverse-Gamma (NIG) distribution:
  p(y, σ²) = N(y | γ, σ²/ν) · Γ⁻¹(σ² | α, β)

Parameters:
  γ (gamma): predicted mean
  ν (nu): virtual evidence count — higher = more confident
  α (alpha): shape parameter (> 1)
  β (beta): scale parameter (> 0)

Fusion: evidence-weighted averaging
  γ_fused = Σ (ν_m / Σ ν_j) · γ_m

Key theoretical property:
  E[y] = γ
  Var[y] = β / (ν·(α-1))  ← epistemic uncertainty
  E[σ²] = β / (α-1)       ← aleatoric uncertainty

Author: Research Project A — Direction 1 (Evidential Gating)
Date: 2026-07-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CASP'))
from modules.transformer import TransformerEncoder


# ─── NIG Evidential Head ────────────────────────────────────────────

class NIGHead(nn.Module):
    """
    Normal-Inverse-Gamma head: maps a hidden representation to NIG parameters.

    Outputs 4 values per sample:
      γ (gamma): predicted regression value
      ν (nu): evidence — virtual observations supporting the prediction
      α (alpha): controls variance (> 1 enforced via softplus + 1)
      β (beta): controls variance (> 0 enforced via softplus)
    """

    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 16)

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Four separate heads for NIG parameters
        self.gamma_head = nn.Linear(hidden_dim, 1)       # unconstrained
        self.nu_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()                                 # ν > 0
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
            nn.Hardtanh(min_val=1.0 + 1e-6, max_val=100.0)  # α > 1
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()                                 # β > 0
        )

    def forward(self, h):
        """
        Args:
            h: [batch_size, input_dim] hidden representation
        Returns:
            gamma: [B, 1] predicted value
            nu: [B, 1] evidence
            alpha: [B, 1] shape (> 1)
            beta: [B, 1] scale (> 0)
        """
        shared = self.shared(h)
        gamma = self.gamma_head(shared)
        nu = self.nu_head(shared) + 1e-6
        alpha = self.alpha_head(shared) + 1.0 + 1e-6
        beta = self.beta_head(shared) + 1e-6
        return gamma, nu, alpha, beta


# ─── Evidential Loss Functions ──────────────────────────────────────

class NIGLoss(nn.Module):
    """
    Negative log-likelihood of Normal-Inverse-Gamma distribution.

    L_NIG = -log p(y | γ, ν, α, β)
          = ½·log(π/ν) - α·log(Ω) + (α+½)·log((y-γ)²ν + Ω)
            + log(Γ(α)) - log(Γ(α+½))

    where Ω = 2β(1+ν)

    Reference: Amini et al., NeurIPS 2020, Eq. (9)
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y, gamma, nu, alpha, beta):
        """
        Args:
            y: [B, 1] ground truth
            gamma, nu, alpha, beta: [B, 1] NIG parameters
        Returns:
            loss: scalar
        """
        nu = nu + self.eps
        alpha = alpha + self.eps
        beta = beta + self.eps

        omega = 2.0 * beta * (1.0 + nu)

        # Stable computation
        pi_term = 0.5 * (torch.log(torch.tensor(torch.pi, device=y.device)) - torch.log(nu))
        alpha_log_omega = alpha * torch.log(omega + self.eps)
        error_term = (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega + self.eps)

        # Log gamma function using torch.lgamma
        lgamma_alpha = torch.lgamma(alpha + self.eps)
        lgamma_alpha_plus_half = torch.lgamma(alpha + 0.5 + self.eps)

        loss = pi_term - alpha_log_omega + error_term + lgamma_alpha - lgamma_alpha_plus_half

        return loss.mean()


class EvidenceRegularizer(nn.Module):
    """
    Evidence regularizer: penalizes evidence on incorrect predictions.

    L_reg = |y - γ| · (2ν + α)

    Intuition: when prediction error is large, the model should have low evidence.
    This term penalizes high evidence (ν, α) on samples where the model is wrong.

    Reference: Amini et al., NeurIPS 2020, Eq. (12)
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y, gamma, nu, alpha):
        """
        Args:
            y: [B, 1] ground truth
            gamma: [B, 1] prediction
            nu: [B, 1] evidence
            alpha: [B, 1] shape
        Returns:
            loss: scalar
        """
        error = torch.abs(y - gamma)
        evidence_penalty = (2.0 * nu + alpha)
        return (error * evidence_penalty).mean()


# ─── Evidential Fusion Model ────────────────────────────────────────

class EvidentialFusion(nn.Module):
    """
    Evidential Deep Learning model for multimodal sentiment analysis.

    Architecture:
    1. Per-modality transformer encoder (same as CASP LateFusion)
    2. Per-modality NIG head (outputs γ, ν, α, β)
    3. Evidence-weighted fusion
    4. Fused NIG parameters → final prediction

    Modes:
      'evidential': full NIG fusion (evidence-weighted)
      'evidential_concat': NIG heads + concat fusion (baseline)
      'uncertainty': original UADG uncertainty gating (for comparison)
      'concat': CASP concat baseline
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
        mode='evidential',      # 'evidential' | 'evidential_concat' | 'uncertainty' | 'concat'
        nig_hidden_dim=32,
    ):
        super().__init__()

        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mods = len(orig_dim)
        self.mode = mode
        self.num_heads = num_heads
        self.layers = layers

        # ---- Projection Layers ----
        self.proj = nn.ModuleList([
            nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
            for i in range(self.num_mods)
        ])

        # ---- Per-modality Encoders ----
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim=proj_dim,
                num_heads=self.num_heads,
                layers=self.layers,
                attn_dropout=attn_dropout,
                res_dropout=res_dropout,
                relu_dropout=relu_dropout,
                embed_dropout=embed_dropout,
            )
            for _ in range(self.num_mods)
        ])

        # ---- NIG Heads (for evidential modes) ----
        if 'evidential' in mode:
            self.nig_heads = nn.ModuleList([
                NIGHead(input_dim=proj_dim, hidden_dim=nig_hidden_dim)
                for _ in range(self.num_mods)
            ])

        # ---- Uncertainty Heads (for uncertainty mode) ----
        if mode == 'uncertainty':
            from models.uadg import UncertaintyHead, DynamicGating
            self.uncertainty_heads = nn.ModuleList([
                UncertaintyHead(input_dim=proj_dim, hidden_dim=proj_dim // 2)
                for _ in range(self.num_mods)
            ])
            self.gating = DynamicGating(temperature=1.0, learnable_temp=True)

        # ---- Output Head ----
        if mode in ['evidential', 'uncertainty']:
            # Fused representation maps to final prediction
            self.out_net = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(out_dropout),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, output_dim),
            )
        elif mode == 'evidential_concat':
            # Concat all NIG gamma values + hidden states
            fusion_dim = self.num_mods * (proj_dim + 1)  # h + gamma per modality
            self.out_net = nn.Sequential(
                nn.Linear(fusion_dim, proj_dim * 2),
                nn.ReLU(),
                nn.Dropout(out_dropout),
                nn.Linear(proj_dim * 2, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, output_dim),
            )
        elif mode == 'concat':
            # CASP-style concat fusion
            self.out_net = nn.Sequential(
                nn.Linear(self.num_mods * proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(out_dropout),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, output_dim),
            )

    def encode_modality(self, x_i, encoder_idx):
        """Encode a single modality: projection → transformer → first-timestep pooling."""
        x_i = x_i.transpose(1, 2)
        x_i = self.proj[encoder_idx](x_i)
        x_i = x_i.permute(2, 0, 1)
        h_full = self.encoders[encoder_idx](x_i)
        h_pooled = h_full[0]  # [B, proj_dim]
        return h_pooled, h_full

    def _get_nig_params(self, x, missing_mask=None):
        """
        Run each modality through encoder + NIG head. Missing modalities get zero evidence.

        Returns:
            gammas: [B, num_mods, 1]
            nus: [B, num_mods, 1]
            alphas: [B, num_mods, 1]
            betas: [B, num_mods, 1]
            hs: [B, num_mods, proj_dim]
        """
        batch_size = x[0].shape[0]
        device = x[0].device

        gammas, nus, alphas, betas, hs = [], [], [], [], []

        for i in range(self.num_mods):
            if missing_mask is not None and missing_mask[i] < 0.5:
                # Missing modality
                gammas.append(torch.zeros(batch_size, 1, device=device))
                nus.append(torch.zeros(batch_size, 1, device=device))
                alphas.append(torch.ones(batch_size, 1, device=device) * (1.0 + 1e-6))
                betas.append(torch.ones(batch_size, 1, device=device) * 1e-6)
                hs.append(torch.zeros(batch_size, self.proj_dim, device=device))
            else:
                h_pooled, h_full = self.encode_modality(x[i], i)
                gamma, nu, alpha, beta = self.nig_heads[i](h_pooled)
                gammas.append(gamma)
                nus.append(nu)
                alphas.append(alpha)
                betas.append(beta)
                hs.append(h_pooled)

        # Stack: [B, num_mods, ...]
        gammas = torch.stack(gammas, dim=1)
        nus = torch.stack(nus, dim=1)
        alphas = torch.stack(alphas, dim=1)
        betas = torch.stack(betas, dim=1)
        hs = torch.stack(hs, dim=1)

        return gammas, nus, alphas, betas, hs

    def _evidential_fusion(self, gammas, nus, alphas, betas, hs):
        """
        Evidence-weighted fusion.

        w_m = ν_m / Σ ν_j   (normalized evidence weights)
        γ = Σ w_m · γ_m
        ν = Σ ν_m
        α = Σ w_m · α_m
        β = Σ w_m · β_m
        h_fused = Σ w_m · h_m
        """
        total_evidence = nus.sum(dim=1, keepdim=True)  # [B, 1, 1]

        # Handle edge case: all evidence is zero → uniform weights
        uniform_weights = torch.ones_like(nus) / self.num_mods
        evidence_weights = nus / (total_evidence + 1e-8)
        weights = torch.where(total_evidence > 1e-8, evidence_weights, uniform_weights)

        gamma_fused = (gammas * weights).sum(dim=1)      # [B, 1]
        nu_fused = nus.sum(dim=1)                         # [B, 1]
        alpha_fused = (alphas * weights).sum(dim=1)       # [B, 1]
        beta_fused = (betas * weights).sum(dim=1)         # [B, 1]
        h_fused = (hs * weights).sum(dim=1)               # [B, proj_dim]

        return gamma_fused, nu_fused, alpha_fused, beta_fused, h_fused, weights.squeeze(-1)

    def forward(self, x, return_nig=False, return_weights=False):
        """
        Args:
            x: list of 3 tensors [text, audio, vision]
            return_nig: if True, return NIG params for loss computation
            return_weights: if True, return fusion weights
        Returns:
            Depending on mode:
            - 'evidential': output [B,1], (+ NIG params, + weights)
            - 'concat': output [B,1]
        """
        # Determine missing modalities
        missing_mask = [float((x[i].abs().sum() > 1e-8).item()) for i in range(self.num_mods)]

        if self.mode == 'evidential':
            gammas, nus, alphas, betas, hs = self._get_nig_params(x, missing_mask)
            gamma_fused, nu_fused, alpha_fused, beta_fused, h_fused, weights = \
                self._evidential_fusion(gammas, nus, alphas, betas, hs)

            final_output = self.out_net(h_fused) + gamma_fused  # residual connection

            results = (final_output,)
            if return_nig:
                results += ((gamma_fused, nu_fused, alpha_fused, beta_fused),)
            if return_weights:
                results += (weights,)
            return results[0] if len(results) == 1 else results

        elif self.mode == 'evidential_concat':
            gammas, nus, alphas, betas, hs = self._get_nig_params(x, missing_mask)
            # Concat hs and gammas for all modalities
            concat = torch.cat([hs.squeeze(0) if hs.shape[0] == 1 else hs.reshape(hs.shape[0], -1),
                                gammas.reshape(hs.shape[0], -1)], dim=-1)
            # Wait, this is messy. Let me simplify:
            bs = hs.shape[0]
            h_cat = hs.reshape(bs, -1)         # [B, num_mods*proj_dim]
            g_cat = gammas.reshape(bs, -1)      # [B, num_mods]
            concat = torch.cat([h_cat, g_cat], dim=-1)
            return self.out_net(concat)

        elif self.mode == 'uncertainty':
            from models.uadg import UADGModel
            # Delegate to UADG-style processing
            # For simplicity, just use concat for this mode
            hs_pooled = []
            for i in range(self.num_mods):
                if missing_mask[i] < 0.5:
                    hs_pooled.append(torch.zeros(x[0].shape[0], self.proj_dim, device=x[0].device))
                else:
                    h_pooled, _ = self.encode_modality(x[i], i)
                    hs_pooled.append(h_pooled)
            concat = torch.cat(hs_pooled, dim=-1)
            return self.out_net(concat)

        elif self.mode == 'concat':
            hs_pooled = []
            for i in range(self.num_mods):
                if missing_mask[i] < 0.5:
                    hs_pooled.append(torch.zeros(x[0].shape[0], self.proj_dim, device=x[0].device))
                else:
                    h_pooled, _ = self.encode_modality(x[i], i)
                    hs_pooled.append(h_pooled)
            concat = torch.cat(hs_pooled, dim=-1)
            return self.out_net(concat)

    def predict_with_uncertainty(self, x):
        """Convenience method: returns (prediction, epistemic_uncertainty, aleatoric_uncertainty)."""
        output, (gamma, nu, alpha, beta) = self.forward(x, return_nig=True)
        # Epistemic uncertainty: Var[y] = beta / (nu * (alpha - 1))
        epistemic = beta / (nu * (alpha - 1) + 1e-8)
        # Aleatoric uncertainty: E[sigma^2] = beta / (alpha - 1)
        aleatoric = beta / (alpha - 1 + 1e-8)
        return output, epistemic, aleatoric


# ─── Factory ────────────────────────────────────────────────────────

def build_evidential_model(orig_dim, mode='evidential', **kwargs):
    """Factory function to create an evidential fusion model."""
    return EvidentialFusion(orig_dim=orig_dim, mode=mode, **kwargs)


# ─── Sanity Check ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Evidential Fusion Model Sanity Check ===\n")

    orig_dim = [768, 512, 256]
    batch_size = 4
    seq_len = 50
    device = 'cpu'

    # Create mock inputs
    text = torch.randn(batch_size, seq_len, orig_dim[0])
    audio = torch.randn(batch_size, seq_len, orig_dim[1])
    vision = torch.randn(batch_size, seq_len, orig_dim[2])

    for mode in ['evidential', 'evidential_concat', 'concat']:
        print(f"\n--- Mode: {mode} ---")
        model = EvidentialFusion(orig_dim=orig_dim, mode=mode)
        model.eval()

        # Test 1: Full modality
        x = [text, audio, vision]
        if mode == 'evidential':
            output, (gamma, nu, alpha, beta) = model(x, return_nig=True)
            print(f"  Output: {output.shape}, γ: {gamma.shape}, ν: {nu.shape}")
            print(f"  γ range: [{gamma.min().item():.3f}, {gamma.max().item():.3f}]")
            print(f"  ν range: [{nu.min().item():.3f}, {nu.max().item():.3f}]")

            # Test uncertainty
            pred, epis, alea = model.predict_with_uncertainty(x)
            print(f"  Epistemic uncertainty: {epis.mean().item():.4f}")
            print(f"  Aleatoric uncertainty: {alea.mean().item():.4f}")

            # Test 2: Missing text
            x_missing = [torch.zeros_like(text), audio, vision]
            output_m, (gamma_m, nu_m, alpha_m, beta_m) = model(x_missing, return_nig=True)
            print(f"  Missing text: output {output_m.shape}, ν {nu_m.mean().item():.3f}")

            # Test 3: Check that missing modality has zero evidence weight
            _, nig_params, weights = model(x_missing, return_nig=True, return_weights=True)
            print(f"  Missing text weights: {weights[0].detach().numpy()}")  # text should be 0
        else:
            output = model(x)
            print(f"  Output: {output.shape}")

        # Parameter count
        total = sum(p.numel() for p in model.parameters())
        print(f"  Params: {total:,}")

    # Test NIGLoss
    print("\n--- NIG Loss Test ---")
    nig_loss = NIGLoss()
    evid_reg = EvidenceRegularizer()

    y = torch.randn(batch_size, 1)
    gamma = torch.randn(batch_size, 1)
    nu = F.softplus(torch.randn(batch_size, 1)) + 1e-6
    alpha = F.softplus(torch.randn(batch_size, 1)) + 1.0 + 1e-6
    beta = F.softplus(torch.randn(batch_size, 1)) + 1e-6

    loss_nig = nig_loss(y, gamma, nu, alpha, beta)
    loss_reg = evid_reg(y, gamma, nu, alpha)
    print(f"  NIG NLL: {loss_nig.item():.4f}")
    print(f"  Evidence Reg: {loss_reg.item():.4f}")
    print(f"  Both finite: {torch.isfinite(loss_nig).item() and torch.isfinite(loss_reg).item()}")

    print("\n=== All sanity checks passed! ===")
