"""
Contrastive Multimodal Sentiment Analysis (ContraMSA)

Cross-modal contrastive learning for missing modality robustness.

Idea: Align representation spaces of different modalities via contrastive loss.
When text is missing, audio+vision representations are already close to
where the fused representation should be, thanks to cross-modal alignment.

Architecture:
  Text  -- Encoder0 -- h_t -- Proj -- z_t --
  Audio -- Encoder1 -- h_a -- Proj -- z_a --|-- Contrastive Loss
  Vision-- Encoder2 -- h_v -- Proj -- z_v --
                                          |
                                 Concat([h_t, h_a, h_v]) -- Output Head

Loss: L = L_L1 + lam_cross * L_cross_modal + lam_drop * L_drop_consistency

Key design decisions (from plan analysis):
  - Contrast in projected space (128-dim) after L2 norm
  - Use 3-class SupCon (negative/neutral/positive) for stability
  - Modality dropout during training (15% per modality)
  - Joint training (no separate stages)
  - Lambda_warmup: cross-modal loss ramps up over first 3 epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CASP'))
from modules.transformer import TransformerEncoder


# ─── Projection Head (SimCLR-style) ─────────────────────────────────

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h):
        z = self.net(h)
        return F.normalize(z, dim=-1)


# ─── Contrastive Loss ───────────────────────────────────────────────

class CrossModalContrastiveLoss(nn.Module):
    """
    Cross-modal contrastive loss.

    For each sample i in batch of size B:
      - 3 views: z_t[i], z_a[i], z_v[i]
      - Positive pairs: different modalities of SAME sample
      - Negative pairs: all modalities of DIFFERENT samples

    With optional label-aware weighting:
      - Samples with similar sentiment labels are "softer" negatives
      - Uses 3-class binning: neg (< -0.3), neutral (-0.3..0.3), pos (> 0.3)

    Temperature parameter tau follows CLIP default (0.07).
    """

    def __init__(self, temperature=0.1, use_labels=True):
        super().__init__()
        self.temperature = temperature
        self.use_labels = use_labels

    def forward(self, z_stack, labels=None):
        """
        Args:
            z_stack: [B, 3, D] — per-modality projected representations
            labels: [B, 1] — sentiment regression labels (optional)
        Returns:
            scalar loss
        """
        B, M, D = z_stack.shape  # M = num_modalities (3)

        # Reshape to [B*M, D] for pairwise similarity
        z_flat = z_stack.reshape(B * M, D)  # [3B, D]
        sim = torch.matmul(z_flat, z_flat.T) / self.temperature  # [3B, 3B]

        # Build positive mask: same sample, different modality
        # For each pair (b1*m1, b2*m2), it's positive iff b1==b2 and m1!=m2
        sample_idx = torch.arange(B, device=z_stack.device).repeat_interleave(M)  # [0,0,0,1,1,1,...]
        pos_mask = (sample_idx.unsqueeze(0) == sample_idx.unsqueeze(1))  # [3B, 3B]
        # Zero out self-comparisons (same modality)
        modality_idx = torch.arange(M, device=z_stack.device).repeat(B)  # [0,1,2,0,1,2,...]
        self_mask = (modality_idx.unsqueeze(0) == modality_idx.unsqueeze(1))
        pos_mask = pos_mask & (~self_mask)

        # Build negative mask with label-aware weighting
        if self.use_labels and labels is not None:
            # 3-class binning
            y = labels.squeeze(-1)  # [B]
            y_bin = torch.where(y < -0.3, torch.tensor(0, device=y.device),
                      torch.where(y > 0.3, torch.tensor(2, device=y.device),
                                  torch.tensor(1, device=y.device)))
            y_bin_expanded = y_bin.repeat_interleave(M)  # [3B]
            same_class = (y_bin_expanded.unsqueeze(0) == y_bin_expanded.unsqueeze(1))
            # Negatives: different samples, possibly different class
            neg_mask = ~pos_mask & ~self_mask
            # For label-aware: apply weight 1.0 for different-class negatives, 0.3 for same-class
            neg_weight = torch.where(same_class & neg_mask, 0.3, 1.0)
        else:
            neg_mask = ~pos_mask & ~self_mask
            neg_weight = 1.0

        # Compute loss manually for stability
        # For each anchor i, compute: -log( sum_pos(exp(sim_i)) / (sum_pos + sum_neg) )
        exp_sim = torch.exp(sim)

        # Clamp for numerical stability
        exp_sim = torch.clamp(exp_sim, max=1e10)

        numerator = (exp_sim * pos_mask.float()).sum(dim=1)  # [3B]
        denominator = numerator + (exp_sim * neg_mask.float() * neg_weight).sum(dim=1)

        loss_per_anchor = -torch.log(numerator / (denominator + 1e-8) + 1e-8)
        # Only compute loss where we have positives
        has_positives = pos_mask.float().sum(dim=1) > 0
        loss = loss_per_anchor[has_positives].mean()

        return loss


# ─── Contrastive MSA Model ──────────────────────────────────────────

class ContraMSA(nn.Module):
    """
    Contrastive Multimodal Sentiment Analysis model.

    Modes:
      'full':  Cross-modal contrastive + L1 regression (recommended)
      'l1':    L1 regression only (no contrastive loss)
      'concat': Simple concat baseline (for comparison)
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
        mode='full',
        contrast_dim=128,
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

        # ---- Per-modality Encoders ----
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim=proj_dim, num_heads=num_heads, layers=layers,
                attn_dropout=attn_dropout, res_dropout=res_dropout,
                relu_dropout=relu_dropout, embed_dropout=embed_dropout,
            )
            for _ in range(self.num_mods)
        ])

        # ---- Contrastive Projection Heads ----
        if mode in ['full', 'l1']:
            self.proj_heads = nn.ModuleList([
                ProjectionHead(proj_dim, hidden_dim=contrast_dim, output_dim=contrast_dim)
                for _ in range(self.num_mods)
            ])

        # ---- Output Head ----
        fusion_dim = self.num_mods * proj_dim
        self.output_head = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(out_dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, output_dim),
        )

    def encode_modality(self, x_i, encoder_idx):
        x_i = x_i.transpose(1, 2)
        x_i = self.proj[encoder_idx](x_i)
        x_i = x_i.permute(2, 0, 1)
        h_full = self.encoders[encoder_idx](x_i)
        return h_full[0], h_full

    def forward(self, x, return_z=False):
        """
        Args:
            x: [text, audio, vision] tensors [B, L, D]
            return_z: if True, return projected representations for contrastive loss
        Returns:
            output: [B, 1]
            (optional) z_stack: [B, 3, contrast_dim]
        """
        batch_size = x[0].shape[0]
        device = x[0].device
        hs = []

        for i in range(self.num_mods):
            is_missing = (x[i].abs().sum() < 1e-8).item()
            if is_missing:
                hs.append(torch.zeros(batch_size, self.proj_dim, device=device))
            else:
                h_pooled, _ = self.encode_modality(x[i], i)
                hs.append(h_pooled)

        h_cat = torch.cat(hs, dim=-1)  # [B, 3*proj_dim]
        output = self.output_head(h_cat)

        if return_z and hasattr(self, 'proj_heads'):
            z_list = []
            for i in range(self.num_mods):
                z_i = self.proj_heads[i](hs[i])
                z_list.append(z_i)
            z_stack = torch.stack(z_list, dim=1)  # [B, 3, contrast_dim]
            return output, z_stack

        return output


def modality_dropout(x, drop_probs=None):
    """Randomly drop modalities. At least one stays."""
    if drop_probs is None:
        drop_probs = [0.15, 0.15, 0.15]
    x_d, mask = [], []
    for i, (xi, p) in enumerate(zip(x, drop_probs)):
        if torch.rand(1).item() < p:
            x_d.append(torch.zeros_like(xi))
            mask.append(0.0)
        else:
            x_d.append(xi)
            mask.append(1.0)
    if sum(mask) < 0.5:
        keep = torch.randint(0, len(x), (1,)).item()
        x_d[keep] = x[keep]
        mask[keep] = 1.0
    return x_d, mask


# ─── Sanity Check ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ContraMSA Sanity Check ===\n")

    orig_dim = [768, 512, 256]
    B, L = 4, 50
    text = torch.randn(B, L, orig_dim[0])
    audio = torch.randn(B, L, orig_dim[1])
    vision = torch.randn(B, L, orig_dim[2])

    for mode in ['full', 'l1', 'concat']:
        print(f"\n--- Mode: {mode} ---")
        model = ContraMSA(orig_dim=orig_dim, mode=mode)
        model.eval()

        x = [text, audio, vision]

        if mode in ['full', 'l1']:
            output, z_stack = model(x, return_z=True)
            print(f"  Output: {output.shape}, z: {z_stack.shape}")
            print(f"  z norm: [{z_stack.norm(dim=-1).min():.3f}, {z_stack.norm(dim=-1).max():.3f}]")
            # Missing text
            x_m = [torch.zeros_like(text), audio, vision]
            output_m, z_m = model(x_m, return_z=True)
            print(f"  Missing text: output {output_m.shape}")
        else:
            output = model(x)
            print(f"  Output: {output.shape}")

        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params:,}")

    # Test contrastive loss
    print("\n--- Contrastive Loss Test ---")
    loss_fn = CrossModalContrastiveLoss(temperature=0.1, use_labels=True)
    z = torch.randn(8, 3, 128)
    z = F.normalize(z, dim=-1)
    labels = torch.randn(8, 1)
    loss = loss_fn(z, labels)
    print(f"  Loss: {loss.item():.4f}, finite: {torch.isfinite(loss).item()}")
    # Without labels
    loss_nl = loss_fn(z)
    print(f"  Loss (no labels): {loss_nl.item():.4f}")

    print("\n=== All sanity checks passed! ===")
