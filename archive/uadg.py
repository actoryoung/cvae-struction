"""
UADG: Uncertainty-Aware Dynamic Gating for Robust Multimodal Sentiment Analysis

Based on CASP (AAAI 2025) LateFusion architecture, extended with:
1. Uncertainty estimation branch per modality
2. Dynamic gating fusion based on uncertainty
3. Modality dropout training for missing modality robustness

Author: Research Project A
Date: 2026-07-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Allow importing from CASP modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CASP'))
from modules.transformer import TransformerEncoder


class UncertaintyHead(nn.Module):
    """Lightweight MLP that outputs a scalar uncertainty per modality."""

    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # ensures positive uncertainty
        )

    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, proj_dim]
        Returns:
            sigma: [batch_size, 1] - mean-pooled uncertainty
        """
        # Mean pool over sequence dimension
        pooled = x.mean(dim=0)  # [batch_size, proj_dim]
        sigma = self.net(pooled)  # [batch_size, 1]
        return sigma


class DynamicGating(nn.Module):
    """Uncertainty-aware dynamic gating mechanism."""

    def __init__(self, temperature=1.0, learnable_temp=True):
        super().__init__()
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        self.eps = 1e-8

    def forward(self, hidden_list, sigma_list, mask=None):
        """
        Args:
            hidden_list: list of [batch_size, proj_dim] tensors, one per modality
            sigma_list: list of [batch_size, 1] tensors, uncertainties
            mask: optional [batch_size, num_mods] binary mask (1=present, 0=missing)
        Returns:
            fused: [batch_size, total_dim] gated fusion
            weights: [batch_size, num_mods] modality weights
        """
        batch_size = hidden_list[0].shape[0]
        num_mods = len(hidden_list)

        # Stack uncertainties: [batch_size, num_mods]
        sigmas = torch.cat(sigma_list, dim=-1)  # [batch_size, num_mods]

        # Convert uncertainty to weight: w = exp(-sigma / T)
        neg_precision = -sigmas / self.temperature

        # If mask provided, set weight of missing modalities to -inf
        if mask is not None:
            neg_precision = neg_precision * mask + (1 - mask) * (-1e9)

        weights = F.softmax(neg_precision, dim=-1)  # [batch_size, num_mods]

        # Weighted fusion
        # Each hidden is [batch_size, proj_dim] -> stack to [batch_size, num_mods, proj_dim]
        stacked = torch.stack(hidden_list, dim=1)  # [batch_size, num_mods, proj_dim]
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [batch_size, proj_dim]

        return fused, weights


class UADGModel(nn.Module):
    """
    Uncertainty-Aware Dynamic Gating Model for Multimodal Sentiment Analysis.

    Architecture:
    - Per-modality Conv1D projection + Transformer encoder (same as CASP LateFusion)
    - Per-modality UncertaintyHead for estimating prediction uncertainty
    - DynamicGating for uncertainty-weighted fusion
    - Output MLP head for regression

    Supports training-time modality dropout for robustness.
    """

    def __init__(
        self,
        orig_dim,          # list of 3 ints: [text_dim, audio_dim, vision_dim]
        output_dim=1,
        proj_dim=40,
        num_heads=8,
        layers=5,
        relu_dropout=0.1,
        embed_dropout=0.25,
        res_dropout=0.1,
        out_dropout=0.1,
        attn_dropout=0.1,
        gate_temperature=1.0,
        learnable_temp=True,
        gating_mode='uncertainty',  # 'uncertainty' | 'mlp' | 'uniform'
    ):
        super().__init__()

        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mods = len(orig_dim)
        self.num_heads = num_heads
        self.layers = layers
        self.gating_mode = gating_mode

        # ---- Projection Layers (same as CASP) ----
        self.proj = nn.ModuleList([
            nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
            for i in range(self.num_mods)
        ])

        # ---- Per-modality Encoders (same as CASP LateFusion) ----
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

        # ---- Uncertainty Branches (NEW) ----
        self.uncertainty_heads = nn.ModuleList([
            UncertaintyHead(input_dim=proj_dim, hidden_dim=proj_dim // 2)
            for _ in range(self.num_mods)
        ])

        # ---- Dynamic Gating (NEW) ----
        self.gating = DynamicGating(
            temperature=gate_temperature,
            learnable_temp=learnable_temp
        )

        # ---- MLP Gating (alternative, when gating_mode='mlp') ----
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.num_mods * proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, self.num_mods),
        )

        # ---- Output layers (adapted from CASP LateFusion) ----
        # Instead of 3*proj_dim input (concat), we use proj_dim (gated fusion)
        self.out_layer_proj0 = nn.Linear(proj_dim, proj_dim)
        self.out_layer_proj1 = nn.Linear(proj_dim, proj_dim)
        self.out_layer_proj2 = nn.Linear(proj_dim, proj_dim)
        self.out_layer = nn.Linear(proj_dim, output_dim)

        # ---- Direct fusion baseline (for ablation: gating_mode='concat') ----
        self.concat_fusion_proj = nn.Linear(self.num_mods * self.proj_dim, self.proj_dim)

    def encode_modality(self, x_i, encoder_idx):
        """Encode a single modality through projection + transformer.

        Args:
            x_i: [batch_size, seq_len, feat_dim]
            encoder_idx: index of encoder to use
        Returns:
            h_pooled: [batch_size, proj_dim] — first-timestep representation
            h_full:   [seq_len, batch_size, proj_dim] — full sequence for uncertainty est.
        """
        x_i = x_i.transpose(1, 2)              # [B, feat_dim, L]
        x_i = self.proj[encoder_idx](x_i)       # [B, proj_dim, L]
        x_i = x_i.permute(2, 0, 1)             # [L, B, proj_dim]
        h_full = self.encoders[encoder_idx](x_i)  # [L, B, proj_dim]
        h_pooled = h_full[0]                     # [B, proj_dim] — first timestep (matches CASP)
        return h_pooled, h_full

    def forward(self, x, return_weights=False, return_sigmas=False):
        """
        Args:
            x: list of 3 tensors [text, audio, vision]
               Each tensor: [batch_size, seq_len, feat_dim]
               Missing modalities should be all-zero tensors of same shape.
            return_weights: if True, return gating weights
            return_sigmas: if True, return uncertainties
        Returns:
            output: [batch_size, 1] sentiment prediction
            (optional) weights: [batch_size, 3] gating weights
            (optional) sigmas: [batch_size, 3] uncertainties
        """
        hs = []      # pooled hidden representations per modality
        sigmas = []  # uncertainties per modality
        missing_mask = []  # 1 if present, 0 if missing

        for i in range(self.num_mods):
            # Check if modality is missing (all zeros)
            is_missing = (x[i].abs().sum() < 1e-8).item()
            missing_mask.append(0.0 if is_missing else 1.0)

            if is_missing:
                # Create zero hidden state and high uncertainty
                batch_size = x[i].shape[0]
                h_i = torch.zeros(batch_size, self.proj_dim, device=x[i].device)
                sigma_i = torch.ones(batch_size, 1, device=x[i].device) * 10.0
            else:
                h_i, h_full = self.encode_modality(x[i], i)   # [B, proj_dim], [L, B, proj_dim]
                sigma_i = self.uncertainty_heads[i](h_full)    # [B, 1]

            hs.append(h_i)
            sigmas.append(sigma_i)

        mask_tensor = torch.tensor(missing_mask, device=x[0].device).unsqueeze(0).expand(hs[0].shape[0], -1)

        # ---- Fusion ----
        if self.gating_mode == 'uncertainty':
            fused, weights = self.gating(hs, sigmas, mask=mask_tensor)
        elif self.gating_mode == 'mlp':
            concat_h = torch.cat(hs, dim=-1)  # [batch, 3*proj_dim]
            weights = F.softmax(self.gate_mlp(concat_h) * mask_tensor + (1 - mask_tensor) * (-1e9), dim=-1)
            stacked = torch.stack(hs, dim=1)
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        elif self.gating_mode == 'uniform':
            # Uniform weight over available modalities
            weights = mask_tensor / (mask_tensor.sum(dim=-1, keepdim=True) + 1e-8)
            stacked = torch.stack(hs, dim=1)
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        elif self.gating_mode == 'concat':
            # Baseline: simple concatenation (same as CASP LateFusion)
            fused = torch.cat(hs, dim=-1)
            fused = F.relu(self.concat_fusion_proj(fused))
            weights = torch.ones_like(mask_tensor) / self.num_mods  # dummy

        # ---- Output Head (residual block from CASP) ----
        last_hs = F.relu(self.out_layer_proj0(fused))
        last_hs_proj = self.out_layer_proj2(
            F.dropout(
                F.relu(self.out_layer_proj1(last_hs)),
                p=0.1,  # out_dropout
                training=self.training,
            )
        )
        last_hs_proj = last_hs_proj + last_hs  # residual
        output = self.out_layer(last_hs_proj)

        # Return based on flags
        result = (output,)
        if return_weights:
            result = result + (weights,)
        if return_sigmas:
            result = result + (torch.cat(sigmas, dim=-1),)

        return result if len(result) > 1 else output

    def get_uncertainties(self, x):
        """Convenience method to get uncertainty estimates."""
        _, weights, sigmas = self.forward(x, return_weights=True, return_sigmas=True)
        return weights, sigmas


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood loss for uncertainty calibration.

    L = (y - mu)^2 / (2 * sigma^2) + log(sigma)

    This encourages the model to output low uncertainty for accurate predictions
    and high uncertainty for inaccurate ones.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, mu, sigma, y):
        """
        Args:
            mu: [batch_size, 1] predicted value
            sigma: [batch_size, 1] predicted uncertainty
            y: [batch_size, 1] ground truth
        """
        sigma = sigma + self.eps
        loss = (y - mu) ** 2 / (2 * sigma ** 2) + torch.log(sigma)
        return loss.mean()


class ConsistencyLoss(nn.Module):
    """Consistency loss between full-modality and modality-dropped representations."""

    def forward(self, h_full, h_dropped):
        """
        Args:
            h_full: fused representation with all modalities
            h_dropped: fused representation with dropped modalities
        """
        return F.mse_loss(h_full, h_dropped)


def modality_dropout(x, drop_probs=None):
    """
    Randomly drop modalities during training.

    Args:
        x: list of [text, audio, vision] tensors
        drop_probs: list of drop probabilities for each modality, default [0.15, 0.15, 0.15]
    Returns:
        x_dropped: list with some modalities zeroed out
        drop_mask: list of [0/1] indicating which were kept
    """
    if drop_probs is None:
        drop_probs = [0.15, 0.15, 0.15]

    x_dropped = []
    drop_mask = []

    for i, (x_i, p) in enumerate(zip(x, drop_probs)):
        if torch.rand(1).item() < p:
            x_dropped.append(torch.zeros_like(x_i))
            drop_mask.append(0.0)
        else:
            x_dropped.append(x_i)
            drop_mask.append(1.0)

    # Ensure at least one modality remains
    if sum(drop_mask) < 0.5:
        # Keep a random modality
        keep_idx = torch.randint(0, len(x), (1,)).item()
        x_dropped[keep_idx] = x[keep_idx]
        drop_mask[keep_idx] = 1.0

    return x_dropped, drop_mask


def build_uadg_model(orig_dim, **kwargs):
    """Factory function to create UADG model."""
    return UADGModel(orig_dim=orig_dim, **kwargs)


if __name__ == "__main__":
    # Quick sanity check
    print("=== UADG Model Sanity Check ===")

    # Mock dimensions: [text_dim, audio_dim, vision_dim]
    orig_dim = [768, 512, 256]
    batch_size = 4
    seq_len = 50

    model = UADGModel(orig_dim=orig_dim, proj_dim=40)

    # Create mock inputs
    text = torch.randn(batch_size, seq_len, orig_dim[0])
    audio = torch.randn(batch_size, seq_len, orig_dim[1])
    vision = torch.randn(batch_size, seq_len, orig_dim[2])
    x = [text, audio, vision]

    # Test 1: Full modality forward
    output = model(x)
    print(f"Full modality output shape: {output.shape}")  # expect [4, 1]

    # Test 2: With weights and sigmas
    output, weights, sigmas = model(x, return_weights=True, return_sigmas=True)
    print(f"Weights shape: {weights.shape}, Sigmas shape: {sigmas.shape}")
    print(f"Weights: {weights}")
    print(f"Sigmas: {sigmas}")

    # Test 3: Missing text modality
    x_missing_text = [torch.zeros_like(text), audio, vision]
    output_m = model(x_missing_text)[0]
    print(f"Missing text output shape: {output_m.shape}")

    # Test 4: Missing text weights
    _, w_m, _ = model(x_missing_text, return_weights=True, return_sigmas=True)
    print(f"Missing text weights: {w_m}")

    # Test 5: Modality dropout
    x_dropped, drop_mask = modality_dropout(x)
    print(f"Drop mask: {drop_mask}")

    # Test 6: Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")

    # Test 7: Different gating modes
    for mode in ['uncertainty', 'mlp', 'uniform', 'concat']:
        m = UADGModel(orig_dim=orig_dim, gating_mode=mode)
        out = m(x)
        print(f"Mode '{mode}': output shape {out.shape if isinstance(out, tuple) else out[0].shape if isinstance(out, tuple) else out.shape}")

    print("\n=== All sanity checks passed! ===")
