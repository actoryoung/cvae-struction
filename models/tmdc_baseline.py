"""
TMDC Baseline (AAAI 2026): Two-stage Modality Denoising and Complementation.

"TMDC: Two-stage Modality Denoising and Complementation for
Incomplete Multimodal Learning." AAAI 2026.

Core idea:
  Stage 1 — Denoising: modality-specific encoders + shared cross-modal
             denoiser extract clean modality-unique and modality-common features.
  Stage 2 — Complementation: when a modality is missing, use the shared
             common features to supplement (reconstruct) the missing one.

Adapted to CASP encoder backbone with classical features (GloVe/COVAREP/FACET).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CASP'))
from modules.transformer import TransformerEncoder


class CrossModalDenoiser(nn.Module):
    """
    Shared cross-modal denoising module (TMDC Stage 1 — common part).
    Takes all available modality embeddings, applies lightweight cross-modal
    self-attention, and produces a modality-common representation.
    """

    def __init__(self, dim=40, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # Projection to common space
        self.common_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, hs_available):
        """
        Args:
            hs_available: list of [B, dim] modality embeddings (2 or 3 tensors)
        Returns:
            h_common: [B, dim] shared common representation
        """
        # Stack as [num_avail, B, dim] for MHA
        h_stack = torch.stack(hs_available, dim=0)  # [K, B, 40]
        attn_out, _ = self.attn(h_stack, h_stack, h_stack)  # [K, B, 40]
        h_norm = self.norm(h_stack + self.dropout(attn_out))
        # Pool across modalities → [B, 40]
        h_pooled = h_norm.mean(dim=0)
        return self.common_proj(h_pooled)


class ComplementationModule(nn.Module):
    """
    TMDC Stage 2 — Complementation.
    Reconstructs a missing modality's embedding from the shared common
    representation.
    """

    def __init__(self, dim=40, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, h_common, target_dim):
        """h_common: [B, dim] → h_complement: [B, target_dim]"""
        h_full = self.mlp(h_common)  # [B, dim]
        return h_full[:, :target_dim]


class TMDC_MSA(nn.Module):
    """
    TMDC: Two-stage Modality Denoising and Complementation for MSA.

    Stage 1: CASP encoders (modality-specific) + CrossModalDenoiser (shared).
    Stage 2: Complementation module fills in missing modalities.
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
        self.max_proj = max(proj_dims)

        # ── Stage 1a: Modality-specific encoders (CASP) ──
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

        # ── Stage 1b: Shared cross-modal denoiser ──
        self.common_denoiser = CrossModalDenoiser(
            dim=self.max_proj, num_heads=4, dropout=out_dropout)

        # ── Stage 2: Complementation ──
        self.complement = ComplementationModule(
            dim=self.max_proj, hidden_dim=64)

        # ── Output Head ──
        self.output_head = nn.Sequential(
            nn.Linear(self.total_proj, self.total_proj * 2 // 3),
            nn.ReLU(),
            nn.Dropout(out_dropout),
            nn.Linear(self.total_proj * 2 // 3, self.total_proj // 3),
            nn.ReLU(),
            nn.Linear(self.total_proj // 3, output_dim),
        )

    def _pad_to_max(self, h):
        if h.shape[-1] < self.max_proj:
            pad = torch.zeros(h.shape[0], self.max_proj - h.shape[-1],
                             device=h.device, dtype=h.dtype)
            return torch.cat([h, pad], dim=-1)
        return h

    def encode_modality(self, x_i, encoder_idx):
        x_i = x_i.transpose(1, 2)
        x_i = self.proj[encoder_idx](x_i)
        x_i = x_i.permute(2, 0, 1)
        seq_out = self.encoders[encoder_idx](x_i)  # CASP returns tensor
        return seq_out[0]  # [B, dim] — first seq position

    def forward(self, x):
        """
        Full forward with TMDC denoising + complementation for missing modalities.

        Args:
            x: [text, audio, vision]

        Returns:
            output: [B, 1]
            h_fused: [B, total_proj]
        """
        batch_size = x[0].shape[0]
        device = x[0].device

        # Stage 1a: Modality-specific encoding
        hs = []
        is_present = []
        for i in range(self.num_mods):
            if x[i].abs().sum() < 1e-8:
                hs.append(None)
                is_present.append(False)
            else:
                h = self.encode_modality(x[i], i)
                hs.append(h)
                is_present.append(True)

        # Stage 1b: Cross-modal common denoising
        available_hs = [hs[i] for i in range(self.num_mods) if is_present[i]]
        if len(available_hs) >= 2:
            h_common = self.common_denoiser(available_hs)  # [B, max_proj]
        elif len(available_hs) == 1:
            h_common = self._pad_to_max(available_hs[0])
        else:
            h_common = torch.zeros(batch_size, self.max_proj, device=device)

        # Stage 2: Complementation for missing modalities
        for i in range(self.num_mods):
            if not is_present[i]:
                hs[i] = self.complement(h_common, self.proj_dims[i])

        # Fill any remaining None
        for i in range(self.num_mods):
            if hs[i] is None:
                hs[i] = torch.zeros(batch_size, self.proj_dims[i], device=device)

        h_cat = torch.cat(hs, dim=-1)
        output = self.output_head(h_cat)
        return output, h_cat

    def forward_with_dropout(self, x, drop_idx):
        """
        Training forward with one modality dropped.
        Returns (output_drop, h_complement, h_true, h_common)
        for MSE reconstruction loss.
        """
        batch_size = x[0].shape[0]
        device = x[0].device

        # Encode all (teacher forcing)
        hs_true = []
        for i in range(self.num_mods):
            h = self.encode_modality(x[i], i)
            hs_true.append(h)

        h_missing_true = hs_true[drop_idx]

        # Stage 1b: Common denoising from available modalities only
        available_hs = [hs_true[i] for i in range(self.num_mods) if i != drop_idx]
        h_common = self.common_denoiser(available_hs)  # [B, max_proj]

        # Stage 2: Complementation
        h_complement = self.complement(h_common, self.proj_dims[drop_idx])

        # Build final h list with complementation
        hs_final = []
        for i in range(self.num_mods):
            if i == drop_idx:
                hs_final.append(h_complement)
            else:
                hs_final.append(hs_true[i])

        h_cat = torch.cat(hs_final, dim=-1)
        output = self.output_head(h_cat)

        return output, h_complement, h_missing_true, h_common


# ─── Sanity Check ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== TMDC Baseline Sanity Check ===\n")

    B, L = 4, 50
    orig_dim = [768, 25, 171]
    text = torch.randn(B, L, 768)
    audio = torch.randn(B, L, 25)
    vision = torch.randn(B, L, 171)

    model = TMDC_MSA(orig_dim=orig_dim, proj_dims=[40, 40, 40])
    model.train()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Full forward
    out, h = model([text, audio, vision])
    print(f"Full: output {out.shape}, h {h.shape}")

    # Missing text
    out_mt, h_mt = model([torch.zeros_like(text), audio, vision])
    print(f"Miss-T: output {out_mt.shape}")

    # Training with dropout
    out_d, h_c, h_t, h_comm = model.forward_with_dropout(
        [text, audio, vision], drop_idx=0)
    print(f"Drop text: out {out_d.shape}, h_complement {h_c.shape}, "
          f"h_true {h_t.shape}, h_common {h_comm.shape}")

    recon_loss = F.mse_loss(h_c, h_t)
    print(f"Recon loss: {recon_loss:.4f}")

    # Component params
    enc_p = sum(p.numel() for p in list(model.proj.parameters()) +
                list(model.encoders.parameters()))
    denoiser_p = sum(p.numel() for p in model.common_denoiser.parameters())
    comp_p = sum(p.numel() for p in model.complement.parameters())
    out_p = sum(p.numel() for p in model.output_head.parameters())
    print(f"Encoders: {enc_p:,} | Denoiser: {denoiser_p:,} | "
          f"Complement: {comp_p:,} | Output: {out_p:,}")

    print("\n=== All checks passed! ===")
