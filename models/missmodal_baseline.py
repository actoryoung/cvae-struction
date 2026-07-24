"""
MissModal Baseline (TACL 2023): Representation Alignment for Missing Modality Robustness.

Ronghao Lin, Haifeng Hu. "MissModal: Increasing Robustness to Missing Modality
in Multimodal Sentiment Analysis." TACL 2023.

Core idea: Instead of generating missing modalities, use three alignment losses
to make missing-modal representations resemble complete-modal representations:
  1. Geometric Contrastive Loss: NT-Xent between complete and missing representations
  2. Distribution Distance Loss: L2 between batch means of complete and missing
  3. Sentiment Semantic Loss: softened KL between predictions

We adapt to CASP encoder backbone (Conv1d + Transformer) with classical features
(GloVe 768d, COVAREP 25d, FACET 171d), identical to our DetMLP/CVAE setup.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CASP'))
from modules.transformer import TransformerEncoder


class MissModal_MSA(nn.Module):
    """
    MSA model trained with MissModal alignment losses.

    Architecture: identical to CASP LateFusion (concat baseline).
    Training: adds contrastive + distribution + semantic alignment losses
    between complete-modal and missing-modal representations.
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
        # MissModal loss hyperparams
        alpha=0.5,      # contrastive loss weight
        beta=0.3,       # distribution loss weight
        gamma=0.5,      # sentiment loss weight
        temperature=0.07,  # contrastive temperature
        sentiment_temp=2.0,  # sentiment KL temperature
        representation_dim=80,  # projection dim for contrastive head
    ):
        super().__init__()
        self.orig_dim = orig_dim
        self.num_mods = len(orig_dim)
        if proj_dims is None:
            proj_dims = [proj_dim] * self.num_mods
        self.proj_dims = proj_dims
        self.total_proj = sum(proj_dims)

        # Loss weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.sentiment_temp = sentiment_temp

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

        # ── Fusion & Output (same as CASP LateFusion) ──
        self.output_head = nn.Sequential(
            nn.Linear(self.total_proj, self.total_proj * 2 // 3),
            nn.ReLU(),
            nn.Dropout(out_dropout),
            nn.Linear(self.total_proj * 2 // 3, self.total_proj // 3),
            nn.ReLU(),
            nn.Linear(self.total_proj // 3, output_dim),
        )

        # ── MissModal contrastive projection head ──
        # Projects fused representation to lower-dim for NT-Xent
        self.contrast_head = nn.Sequential(
            nn.Linear(self.total_proj, representation_dim),
            nn.ReLU(),
            nn.Linear(representation_dim, representation_dim),
        )

    def encode_modality(self, x_i, encoder_idx):
        x_i = x_i.transpose(1, 2)
        x_i = self.proj[encoder_idx](x_i)
        x_i = x_i.permute(2, 0, 1)
        h_full = self.encoders[encoder_idx](x_i)
        return h_full[0], h_full

    def _encode_all(self, x):
        """Encode all modalities, return pooled representations."""
        hs = []
        for i in range(self.num_mods):
            h_pooled, _ = self.encode_modality(x[i], i)
            hs.append(h_pooled)
        return hs

    def _fuse_and_predict(self, hs):
        """Concat fusion → prediction. hs may contain None for missing."""
        device = hs[0].device if hs[0] is not None else next(
            h.device for h in hs if h is not None)
        batch_size = next(h.shape[0] for h in hs if h is not None)
        filled = []
        for i, h in enumerate(hs):
            if h is None:
                filled.append(torch.zeros(batch_size, self.proj_dims[i], device=device))
            else:
                filled.append(h)
        h_cat = torch.cat(filled, dim=-1)
        return self.output_head(h_cat), h_cat

    def forward(self, x):
        """Full forward: encode all → fuse → predict."""
        hs = self._encode_all(x)
        output, h_fused = self._fuse_and_predict(hs)
        return output, h_fused

    def forward_missing(self, x, drop_idx):
        """Forward with one modality dropped. Returns (output, h_fused, h_complete)."""
        hs = self._encode_all(x)
        # Complete representation (for alignment)
        _, h_complete = self._fuse_and_predict(hs)
        # Drop one
        hs_miss = hs.copy()
        hs_miss[drop_idx] = None
        output_drop, h_fused_drop = self._fuse_and_predict(hs_miss)
        return output_drop, h_fused_drop, h_complete

    def geometric_contrastive_loss(self, h_complete, h_missing):
        """
        NT-Xent contrastive loss between complete and missing representations.
        h_complete: [B, D] fused representation from full modalities
        h_missing:  [B, D] fused representation with one modality dropped
        Positive pairs: (h_complete_i, h_missing_i) — same sample
        Negative pairs: all other pairs in the batch
        """
        z_c = F.normalize(self.contrast_head(h_complete), dim=-1)
        z_m = F.normalize(self.contrast_head(h_missing), dim=-1)

        # Compute similarity matrix: cosine sim between all complete and missing pairs
        # sim[i,j] = z_c_i · z_m_j / temperature
        sim = z_c @ z_m.T / self.temperature  # [B, B]

        # Positive pairs are on diagonal (same sample)
        B = sim.shape[0]
        labels = torch.arange(B, device=sim.device)

        # Symmetric loss: row-wise + column-wise
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss

    def distribution_distance_loss(self, h_complete, h_missing):
        """
        L2 distance between batch-level means of complete and missing representations.
        Encourages the distribution of missing-modal representations to match complete.
        """
        mu_c = h_complete.mean(dim=0)
        mu_m = h_missing.mean(dim=0)
        return F.mse_loss(mu_m, mu_c)  # L2^2 between means

    def sentiment_semantic_loss(self, y_complete, y_missing):
        """
        Softened KL divergence between predictions from complete and missing inputs.
        Uses temperature to soften distributions before KL.
        y_complete: [B, 1] regression predictions from full input
        y_missing:  [B, 1] regression predictions with dropped modality
        """
        # For regression (single output), use MSE with temperature scaling
        # Original MissModal uses KL for classification; we adapt as softened MSE
        soft_c = y_complete / self.sentiment_temp
        soft_m = y_missing / self.sentiment_temp
        return F.mse_loss(soft_m, soft_c)


def missmodal_train_step(model, x, labels, drop_idx, l1_loss_fn, optimizer=None):
    """
    One training step with MissModal alignment losses.

    Returns:
        total_loss, reg_full, reg_drop, loss_contrast, loss_dist, loss_sentiment
    """
    # Full modality forward
    output_full, h_full = model(x)
    reg_full = l1_loss_fn(output_full, labels)

    # Missing modality forward
    output_drop, h_drop, h_complete = model.forward_missing(x, drop_idx)
    reg_drop = l1_loss_fn(output_drop, labels)

    # Alignment losses
    loss_contrast = model.geometric_contrastive_loss(h_complete, h_drop)
    loss_dist = model.distribution_distance_loss(h_complete, h_drop)
    loss_sentiment = model.sentiment_semantic_loss(output_full, output_drop)

    total = (reg_full + reg_drop +
             model.alpha * loss_contrast +
             model.beta * loss_dist +
             model.gamma * loss_sentiment)

    return total, reg_full, reg_drop, loss_contrast, loss_dist, loss_sentiment


# ─── Sanity Check ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== MissModal Baseline Sanity Check ===\n")
    B, L = 4, 50
    orig_dim = [768, 25, 171]
    text = torch.randn(B, L, 768)
    audio = torch.randn(B, L, 25)
    vision = torch.randn(B, L, 171)
    labels = torch.randn(B, 1)

    model = MissModal_MSA(orig_dim=orig_dim, proj_dims=[40, 40, 40])
    model.train()

    # Full forward
    out_full, h_f = model([text, audio, vision])
    print(f"Full: output {out_full.shape}, h_fused {h_f.shape}")

    # Missing text forward
    out_drop, h_drop, h_c = model.forward_missing([text, audio, vision], drop_idx=0)
    print(f"Drop text: output {out_drop.shape}, h_drop {h_drop.shape}")

    # Alignment losses
    lc = model.geometric_contrastive_loss(h_c, h_drop)
    ld = model.distribution_distance_loss(h_c, h_drop)
    ls = model.sentiment_semantic_loss(out_full, out_drop)
    print(f"\nLosses: contrast={lc:.4f}, dist={ld:.6f}, sentiment={ls:.4f}")

    # Full training step
    total, rf, rd, lc, ld, ls = missmodal_train_step(
        model, [text, audio, vision], labels, drop_idx=0,
        l1_loss_fn=nn.L1Loss())
    print(f"\nTraining step: total={total:.4f}, reg_full={rf:.4f}, "
          f"reg_drop={rd:.4f}, contr={lc:.4f}, dist={ld:.6f}, sent={ls:.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {params:,}")
    print("=== All checks passed! ===")
