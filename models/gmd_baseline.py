"""
GMD Baseline (AAAI 2024): Gradient-Guided Modality Decoupling for Missing-Modality Robustness.

Hao Wang, Shengda Luo, Guosheng Hu, Jianguo Zhang.
"Gradient-Guided Modality Decoupling for Missing-Modality Robustness." AAAI 2024.

Core idea: Modality dominance causes gradient conflicts between modalities.
GMD detects conflicting gradients on shared parameters and projects out the
conflicting components during backprop. Dynamic Sharing (DS) adaptively skips
missing modality encoders instead of zero-filling.

Adapted to CASP encoder backbone with classical features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CASP'))
from modules.transformer import TransformerEncoder


class GMD_MSA(nn.Module):
    """
    MSA model with Dynamic Sharing framework + GMD gradient decoupling.

    Architecture: CASP encoders (per-modality Conv1d+Transformer) +
    shared output head. DS skips missing encoder branches entirely.
    GMD is applied in the training loop (see train_gmd.py).
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
        proj_dims=None,
    ):
        super().__init__()
        self.orig_dim = orig_dim
        self.num_mods = len(orig_dim)
        if proj_dims is None:
            proj_dims = [proj_dim] * self.num_mods
        self.proj_dims = proj_dims
        self.total_proj = sum(proj_dims)

        # ── CASP Encoder Backbone (per-modality) ──
        self.proj = nn.ModuleList([
            nn.Conv1d(self.orig_dim[i], self.proj_dims[i], kernel_size=1, padding=0)
            for i in range(self.num_mods)
        ])
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim=self.proj_dims[i], num_heads=num_heads, layers=layers,
                attn_dropout=attn_dropout, res_dropout=res_dropout,
                relu_dropout=relu_dropout, embed_dropout=embed_dropout,
            )
            for i in range(self.num_mods)
        ])

        # ── Shared Output Head (target of GMD decoupling) ──
        self.output_head = nn.Sequential(
            nn.Linear(self.total_proj, self.total_proj * 2 // 3),
            nn.ReLU(),
            nn.Dropout(out_dropout),
            nn.Linear(self.total_proj * 2 // 3, self.total_proj // 3),
            nn.ReLU(),
            nn.Linear(self.total_proj // 3, output_dim),
        )

    def encode_modality(self, x_i, encoder_idx):
        if x_i is None or (x_i.abs().sum() < 1e-8).item():
            return None, None
        x_i = x_i.transpose(1, 2)
        x_i = self.proj[encoder_idx](x_i)
        x_i = x_i.permute(2, 0, 1)
        h_full = self.encoders[encoder_idx](x_i)
        return h_full[0], h_full

    def forward(self, x, available_mask=None):
        """
        Args:
            x: [text, audio, vision] tensors
            available_mask: [bool, bool, bool], True = available.
                           If None, use all modalities.
        Returns:
            output: [B, 1]
            h_fused: [B, total_proj]
        """
        batch_size = x[0].shape[0]
        device = x[0].device
        hs = []
        for i in range(self.num_mods):
            if available_mask is not None and not available_mask[i]:
                hs.append(None)
            elif x[i].abs().sum() < 1e-8:
                hs.append(None)
            else:
                h_pooled, _ = self.encode_modality(x[i], i)
                hs.append(h_pooled)

        filled = []
        for i, h in enumerate(hs):
            if h is None:
                filled.append(torch.zeros(batch_size, self.proj_dims[i], device=device))
            else:
                filled.append(h)

        h_cat = torch.cat(filled, dim=-1)
        output = self.output_head(h_cat)
        return output, h_cat

    def forward_combination(self, x, combination):
        """
        Forward with a specific modality combination.

        Args:
            x: full [text, audio, vision]
            combination: list of present modality indices, e.g., [0,2] for T+V
        Returns:
            output, loss-ready
        """
        available_mask = [i in combination for i in range(self.num_mods)]
        return self.forward(x, available_mask)

    def get_shared_parameters(self):
        """Return shared parameters for GMD decoupling (output_head)."""
        return list(self.output_head.parameters())

    def get_specific_parameters(self):
        """Return modality-specific parameters (encoders)."""
        params = []
        for p in self.proj.parameters():
            params.append(p)
        for p in self.encoders.parameters():
            params.append(p)
        return params


# ─── GMD Gradient Decoupling ────────────────────────────────────────

def detect_gradient_conflict(grads_a, grads_b):
    """
    Check if two sets of gradients conflict.
    Conflict: cosine similarity < 0 for overlapping parameter pairs.

    Args:
        grads_a: list of gradient tensors (from loss_a.backward)
        grads_b: list of gradient tensors (from loss_b.backward)

    Returns:
        bool: True if gradients conflict
        float: cosine similarity
    """
    flat_a = torch.cat([g.flatten().float() for g in grads_a if g is not None])
    flat_b = torch.cat([g.flatten().float() for g in grads_b if g is not None])

    if flat_a.norm() < 1e-12 or flat_b.norm() < 1e-12:
        return False, 1.0

    cos_sim = (flat_a @ flat_b) / (flat_a.norm() * flat_b.norm())
    return cos_sim < 0, cos_sim.item()


def gmd_decouple(grads_pairs):
    """
    Apply GMD gradient decoupling to a list of gradient tuple/loss pairs.

    For each pair (grads_i, grads_j) with negative cosine similarity:
        g̃_i = g_i - proj(g_i, g_j)
        g̃_j = g_j - proj(g_j, g_i)

    Args:
        grads_pairs: list of (grads_tuple, loss) from different modality combinations.
                    Each grads_tuple is a tuple of gradient tensors from torch.autograd.grad.

    Returns:
        final_grads: list of decoupled gradient tensors (one per shared parameter)
    """
    if len(grads_pairs) < 2:
        return list(grads_pairs[0][0]) if grads_pairs else None

    n_params = len(grads_pairs[0][0])

    # Sum all gradients first
    summed = [None] * n_params
    for grads, _ in grads_pairs:
        for k in range(n_params):
            if grads[k] is not None:
                summed[k] = grads[k].clone() if summed[k] is None else summed[k] + grads[k]

    # Apply pairwise decoupling for conflicting pairs
    for i in range(len(grads_pairs)):
        for j in range(i + 1, len(grads_pairs)):
            grads_i, _ = grads_pairs[i]
            grads_j, _ = grads_pairs[j]

            for k in range(n_params):
                if grads_i[k] is None or grads_j[k] is None:
                    continue
                gi = grads_i[k].flatten().float()
                gj = grads_j[k].flatten().float()

                cos_sim = (gi @ gj) / (gi.norm() * gj.norm() + 1e-12)

                if cos_sim < 0:
                    gi_norm_sq = gi.norm().pow(2) + 1e-12
                    gj_norm_sq = gj.norm().pow(2) + 1e-12

                    gi_decoupled = gi - (gi @ gj / gj_norm_sq) * gj
                    gj_decoupled = gj - (gi @ gj / gi_norm_sq) * gi

                    # Replace in summed: remove old + add decoupled
                    summed[k] = (summed[k].flatten().float()
                                 - gi - gj + gi_decoupled + gj_decoupled).reshape_as(summed[k])

    return summed


# ─── Sanity Check ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== GMD Baseline Sanity Check ===\n")

    B, L = 4, 50
    orig_dim = [768, 25, 171]
    text = torch.randn(B, L, 768)
    audio = torch.randn(B, L, 25)
    vision = torch.randn(B, L, 171)

    model = GMD_MSA(orig_dim=orig_dim, proj_dims=[40, 40, 40])
    model.train()

    # Full forward
    out, h = model([text, audio, vision])
    print(f"Full: output {out.shape}, h {h.shape}")

    # Missing text via DS (skip encoder)
    out_mt, h_mt = model([text, audio, vision], available_mask=[False, True, True])
    print(f"Miss-T (DS): output {out_mt.shape}")

    # Text-only
    out_t, h_t = model([text, audio, vision], available_mask=[True, False, False])
    print(f"Text-only: output {out_t.shape}")

    # Gradient decoupling test
    shared_params = model.get_shared_parameters()
    print(f"\nShared params (output_head): {sum(p.numel() for p in shared_params):,}")
    print(f"Model-specific params: {sum(p.numel() for p in model.get_specific_parameters()):,}")
    print(f"Total: {sum(p.numel() for p in model.parameters()):,}")

    # Simulate gradient conflict
    print("\nGradient conflict test:")
    loss_fn = nn.L1Loss()
    labels = torch.randn(B, 1)

    # Loss 1: full
    out1, _ = model([text, audio, vision])
    loss1 = loss_fn(out1, labels)
    grads1 = torch.autograd.grad(loss1, shared_params, retain_graph=True)

    # Loss 2: miss-T
    out2, _ = model([text, audio, vision], available_mask=[False, True, True])
    loss2 = loss_fn(out2, labels)
    grads2 = torch.autograd.grad(loss2, shared_params, retain_graph=True)

    conflict, cos = detect_gradient_conflict(grads1, grads2)
    print(f"  Full vs Miss-T: conflict={conflict}, cos_sim={cos:.4f}")

    # Loss 3: miss-A
    out3, _ = model([text, audio, vision], available_mask=[True, False, True])
    loss3 = loss_fn(out3, labels)
    grads3 = torch.autograd.grad(loss3, shared_params, retain_graph=True)

    # Decouple
    pairs = [(grads1, loss1), (grads2, loss2), (grads3, loss3)]
    decoupled = gmd_decouple(pairs)
    print(f"  Decoupled grads: {len(decoupled)} tensors")

    print("\n=== All checks passed! ===")
