"""
MulT: Multimodal Transformer (Tsai et al., ACL 2019)
Cross-modal attention for multimodal fusion.

Implemented within CASP framework for fair comparison:
  - Same Conv1d projections
  - Same evaluation protocol
  - Handles missing modalities via zero-filling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Cross-modal attention: one modality attends to another.

    Source modality provides Q, target modality provides K and V.
    """
    def __init__(self, embed_dim=40, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, src, tgt):
        """src provides Q, tgt provides K and V.
        Args:
            src: [L, B, embed_dim] — source modality (provides Q)
            tgt: [L, B, embed_dim] — target modality (provides K, V)
        Returns:
            [L, B, embed_dim]
        """
        L, B, D = src.shape
        Q = self.q_proj(src).view(L, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        K = self.k_proj(tgt).view(L, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        V = self.v_proj(tgt).view(L, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, H, L, D/H]
        out = out.permute(2, 0, 1, 3).contiguous().view(L, B, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        return self.layer_norm(src + out)


class MulT(nn.Module):
    """Multimodal Transformer (Tsai et al., ACL 2019).

    Processes each pair of modalities through cross-modal attention.
    Handles missing modalities by zero-filling the attention outputs.
    """
    def __init__(self, orig_dim, proj_dim=40, num_heads=8, num_layers=4,
                 attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 embed_dropout=0.25, out_dropout=0.1):
        super().__init__()
        self.orig_dim = orig_dim
        self.num_mods = len(orig_dim)
        self.proj_dim = proj_dim

        # Conv1d projections (same as CASP)
        self.proj = nn.ModuleList([
            nn.Conv1d(orig_dim[i], proj_dim, kernel_size=1, padding=0)
            for i in range(self.num_mods)
        ])

        # Positional embedding
        self.pe = nn.Parameter(torch.randn(1, 200, proj_dim) * 0.02)

        # Embed dropout
        self.embed_dropout = nn.Dropout(embed_dropout)

        # Cross-modal attention layers for each direction (T<->A, T<->V, A<->V)
        self.cross_attn = nn.ModuleDict()
        mod_names = ['T', 'A', 'V']
        for i in range(self.num_mods):
            for j in range(self.num_mods):
                if i != j:
                    key = f"{mod_names[i]}->{mod_names[j]}"
                    self.cross_attn[key] = nn.ModuleList([
                        CrossModalAttention(proj_dim, num_heads, attn_dropout)
                        for _ in range(num_layers)
                    ])

        # Output head
        total_dim = proj_dim * self.num_mods * (self.num_mods - 1)  # 6 × proj_dim
        self.output_head = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(out_dropout),
            nn.Linear(total_dim // 2, total_dim // 4),
            nn.ReLU(),
            nn.Linear(total_dim // 4, 1),
        )

    def forward(self, x):
        """Forward pass with cross-modal attention.
        Args:
            x: [text, audio, vision] each [B, L, D_m]
        Returns:
            output: [B, 1]
        """
        batch_size = x[0].shape[0]
        device = x[0].device
        mod_names = ['T', 'A', 'V']

        # Project each modality
        projected = []
        for i in range(self.num_mods):
            xi = x[i].transpose(1, 2)  # [B, D, L]
            xi = self.proj[i](xi)       # [B, proj_dim, L]
            xi = xi.permute(2, 0, 1)    # [L, B, proj_dim]
            L = xi.shape[0]
            xi = xi + self.pe[:, :L, :].transpose(0, 1).to(device)
            xi = self.embed_dropout(xi)
            projected.append(xi)

        # Cross-modal attention for each direction
        cross_outputs = []
        for i in range(self.num_mods):
            for j in range(self.num_mods):
                if i == j:
                    continue
                key = f"{mod_names[i]}->{mod_names[j]}"
                src = projected[i]  # provides query
                tgt = projected[j]  # provides key/value
                out = src
                for layer in self.cross_attn[key]:
                    out = layer(out, tgt)
                # Pool: take mean over time dimension
                out_pooled = out.mean(dim=0)  # [B, proj_dim]
                cross_outputs.append(out_pooled)

        # Concat all cross-modal outputs and predict
        h_cat = torch.cat(cross_outputs, dim=-1)  # [B, 6*proj_dim]
        return self.output_head(h_cat), None
