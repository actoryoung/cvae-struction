"""
T2DR Baseline (ACL 2025 Findings): Two-Tier Deficiency-Resistant Framework.

Han Lin et al. "T2DR: A Two-Tier Deficiency-Resistant Framework for
Incomplete Multimodal Learning." Findings of ACL 2025.

Core components:
  1. IADR (Intra-Modal Deficiency-Resistant): Intra-Attn on each modality's features
     to focus on available data segments while avoiding suppression of missing regions.
  2. IEDR (Inter-Modal Deficiency-Resistant):
     a. SFP (Shared Feature Prediction): predicts missing modality embeddings
        from available modalities via a shared MLP predictor
     b. CAS (Capability-Aware Scorer): scores each modality's reliability
     c. Inter-Attn: CAS-weighted cross-modal attention for fusion

Adapted to CASP encoder backbone with classical features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CASP'))
from modules.transformer import TransformerEncoder


class IntraModalAttention(nn.Module):
    """
    Intra-Attention module (IADR component).
    Self-attention on each modality's features to learn which temporal
    segments carry useful information, preventing excessive suppression
    of partially-missing regions.
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [seq_len, batch_size, dim] (CASP Transformer output format)
        Returns: [batch_size, dim] pooled
        """
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        # Pool: mean over sequence dimension
        return x.mean(dim=0)  # [B, dim]


class SharedFeaturePredictor(nn.Module):
    """
    SFP: Shared Feature Prediction module (IEDR component).
    Predicts a missing modality's embedding from available modalities.
    Single shared MLP for all modality combinations.
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=40, num_mods=3):
        super().__init__()
        self.num_mods = num_mods
        self.output_dim = output_dim

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h_available_list, missing_idx):
        """
        Args:
            h_available_list: list of available h tensors
            missing_idx: which modality to predict
        Returns:
            h_pred: [B, output_dim] predicted missing modality embedding
        """
        h_concat = torch.cat(h_available_list, dim=-1)
        return self.predictor(h_concat)


class CapabilityAwareScorer(nn.Module):
    """
    CAS: Capability-Aware Scorer (IEDR component).
    Scores each modality's reliability based on its encoded features.
    """

    def __init__(self, dim=40, hidden_dim=32):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h):
        """h: [B, dim] → score: [B, 1]"""
        return self.scorer(h)


class T2DR_MSA(nn.Module):
    """
    T2DR: Two-Tier Deficiency-Resistant MSA model.

    Tier 1 (IADR): Intra-modal self-attention on each modality.
    Tier 2 (IEDR): SFP + CAS + Inter-Attn fusion.
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

        # ── CASP Encoder Backbone ──
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

        # ── Tier 1: Intra-Modal Deficiency-Resistant (IADR) ──
        self.intra_attn = nn.ModuleList([
            IntraModalAttention(dim=self.proj_dims[i], num_heads=4, dropout=out_dropout)
            for i in range(self.num_mods)
        ])

        # ── Tier 2: Inter-Modal Deficiency-Resistant (IEDR) ──
        # SFP: Shared Feature Predictor
        sfp_input_dim = (self.num_mods - 1) * self.max_proj
        self.sfp = SharedFeaturePredictor(
            input_dim=sfp_input_dim, hidden_dim=128,
            output_dim=self.max_proj, num_mods=self.num_mods)

        # CAS: Capability-Aware Scorer (per-modality + unified)
        self.cas = nn.ModuleList([
            CapabilityAwareScorer(dim=self.proj_dims[i], hidden_dim=32)
            for i in range(self.num_mods)
        ])

        # Inter-modal attention (CAS-weighted fusion)
        # MHA expects [seq_len, batch, embed_dim] with batch_first=False
        self.inter_attn = nn.MultiheadAttention(
            embed_dim=self.max_proj, num_heads=4, batch_first=False)

        # ── Output Head ──
        fused_dim = self.num_mods * self.max_proj
        self.output_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim * 2 // 3),
            nn.ReLU(),
            nn.Dropout(out_dropout),
            nn.Linear(fused_dim * 2 // 3, fused_dim // 3),
            nn.ReLU(),
            nn.Linear(fused_dim // 3, output_dim),
        )

    def _pad_to_max(self, h):
        """Pad representation to max_proj dim if needed."""
        if h.shape[-1] < self.max_proj:
            pad = torch.zeros(h.shape[0], self.max_proj - h.shape[-1],
                             device=h.device, dtype=h.dtype)
            return torch.cat([h, pad], dim=-1)
        return h

    def encode_modality(self, x_i, encoder_idx):
        """Encode one modality with projection + Transformer + intra-attention."""
        x_i = x_i.transpose(1, 2)
        x_i = self.proj[encoder_idx](x_i)
        x_i = x_i.permute(2, 0, 1)
        # CASP TransformerEncoder returns tensor [seq_len, batch, dim] directly
        seq_out = self.encoders[encoder_idx](x_i)  # [seq_len, batch, dim]
        # Tier 1: Intra-modal attention → pooled [batch, dim]
        h_intra = self.intra_attn[encoder_idx](seq_out)
        return h_intra

    def forward(self, x):
        """
        Full forward pass through T2DR.

        Args:
            x: [text, audio, vision] tensors (zero-filled if missing)

        Returns:
            output: [B, 1]
            h_fused: [B, total_proj_padded]
        """
        batch_size = x[0].shape[0]
        device = x[0].device

        # Step 1: Encode all modalities (skip zeros)
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

        # Step 2: SFP — predict missing modalities
        available_hs = [hs[i] for i in range(self.num_mods) if is_present[i]]
        missing_indices = [i for i in range(self.num_mods) if not is_present[i]]

        if missing_indices and len(available_hs) > 0:
            # Pad available h's to max_proj for SFP
            avail_padded = [self._pad_to_max(h) for h in available_hs]
            for mi in missing_indices:
                h_pred = self.sfp(avail_padded, mi)  # [B, max_proj]
                # Slice to correct dim
                hs[mi] = h_pred[:, :self.proj_dims[mi]]

        # Fill remaining None
        for i in range(self.num_mods):
            if hs[i] is None:
                hs[i] = torch.zeros(batch_size, self.proj_dims[i], device=device)

        # Step 3: CAS — score each modality
        cas_scores = []
        for i in range(self.num_mods):
            score = self.cas[i](hs[i])  # [B, 1]
            cas_scores.append(score)

        # Step 4: Inter-Attn with CAS weighting
        # Stack to [num_mods, batch, max_proj] for MHA (batch_first=False)
        hs_padded = torch.stack([self._pad_to_max(h) for h in hs], dim=0)  # [3, B, max_proj]
        attn_out, _ = self.inter_attn(hs_padded, hs_padded, hs_padded)      # [3, B, max_proj]

        # Weight by CAS scores
        cas_stack = torch.cat(cas_scores, dim=-1)  # [B, 3]
        cas_weights = F.softmax(cas_stack, dim=-1).unsqueeze(-1)  # [B, 3, 1]
        attn_out = attn_out.transpose(0, 1)  # [B, 3, max_proj]
        h_weighted = (attn_out * cas_weights).reshape(batch_size, -1)  # [B, 3*max_proj]

        # Step 5: Output
        output = self.output_head(h_weighted)
        h_fused = h_weighted

        return output, h_fused

    def forward_with_dropout(self, x, drop_idx):
        """
        Training forward with one modality dropped.
        Returns (output_drop, h_pred, h_true, h_avail_padded)
        for MSE reconstruction loss on SFP.
        """
        batch_size = x[0].shape[0]
        device = x[0].device

        # Encode all
        hs_true = []
        for i in range(self.num_mods):
            h = self.encode_modality(x[i], i)
            hs_true.append(h)

        available_hs = [hs_true[i] for i in range(self.num_mods) if i != drop_idx]
        h_missing_true = hs_true[drop_idx]

        # SFP prediction
        avail_padded = [self._pad_to_max(h) for h in available_hs]
        h_pred = self.sfp(avail_padded, drop_idx)
        h_pred_sliced = h_pred[:, :self.proj_dims[drop_idx]]
        h_avail_padded = torch.cat(avail_padded, dim=-1)

        # ── Full T2DR forward with the dropped modality replaced ──
        hs = hs_true.copy()
        hs[drop_idx] = h_pred_sliced

        # CAS
        cas_scores = [self.cas[i](hs[i]) for i in range(self.num_mods)]
        cas_stack = torch.cat(cas_scores, dim=-1)

        # Inter-Attn with CAS weighting (batch_first=False format)
        hs_padded = torch.stack([self._pad_to_max(h) for h in hs], dim=0)  # [3, B, max_proj]
        attn_out, _ = self.inter_attn(hs_padded, hs_padded, hs_padded)      # [3, B, max_proj]
        cas_weights = F.softmax(cas_stack, dim=-1).unsqueeze(-1)  # [B, 3, 1]
        attn_out_t = attn_out.transpose(0, 1)  # [B, 3, max_proj]
        h_weighted = (attn_out_t * cas_weights).reshape(batch_size, -1)  # [B, 3*max_proj]

        output = self.output_head(h_weighted)

        return output, h_pred_sliced, h_missing_true, h_avail_padded


# ─── Sanity Check ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== T2DR Baseline Sanity Check ===\n")

    B, L = 4, 50
    orig_dim = [768, 25, 171]
    text = torch.randn(B, L, 768)
    audio = torch.randn(B, L, 25)
    vision = torch.randn(B, L, 171)

    model = T2DR_MSA(orig_dim=orig_dim, proj_dims=[40, 40, 40])
    model.train()

    # Full forward
    out, h = model([text, audio, vision])
    print(f"Full: output {out.shape}, h_fused {h.shape}")

    # Missing text (zero-fill input)
    out_mt, h_mt = model([torch.zeros_like(text), audio, vision])
    print(f"Miss-T (zero-fill): output {out_mt.shape}")

    # Missing audio
    out_ma, h_ma = model([text, torch.zeros_like(audio), vision])
    print(f"Miss-A: output {out_ma.shape}")

    # Training with dropout
    out_d, h_pred, h_true, h_avail = model.forward_with_dropout(
        [text, audio, vision], drop_idx=0)
    print(f"\nDrop text train: output {out_d.shape}, h_pred {h_pred.shape}, "
          f"h_true {h_true.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {params:,}")

    # Component breakdown
    enc_params = sum(p.numel() for p in list(model.proj.parameters()) +
                     list(model.encoders.parameters()))
    iadr_params = sum(p.numel() for p in model.intra_attn.parameters())
    sfp_params = sum(p.numel() for p in model.sfp.parameters())
    cas_params = sum(p.numel() for p in model.cas.parameters())
    inter_params = sum(p.numel() for p in model.inter_attn.parameters())
    out_params = sum(p.numel() for p in model.output_head.parameters())
    print(f"  Encoders: {enc_params:,}")
    print(f"  IADR (intra-attn): {iadr_params:,}")
    print(f"  SFP (predictor): {sfp_params:,}")
    print(f"  CAS (scorer): {cas_params:,}")
    print(f"  Inter-Attn: {inter_params:,}")
    print(f"  Output head: {out_params:,}")

    # Verify reconstruction loss shape
    import torch.nn.functional as F
    recon_loss = F.mse_loss(h_pred, h_true)
    print(f"\nReconstruction loss: {recon_loss:.4f}")

    print("\n=== All checks passed! ===")
