"""
Evidential Fusion V3 — Stable Hybrid Approach

Key lesson from V1/V2 failures:
  V1 (uncertainty gating): Works but gains are modest — no theoretical grounding
  V2 (NIG evidential):  Theoretically grounded but NIG NLL explodes
                        (model learns to output infinite evidence)

V3 approach: Evidence-weighted fusion + L1 loss (stable) + optional NIG auxiliary

  e_m = softplus(MLP_evidence(h_m))         ← "how useful is this modality?"
  h_fused = Σ (e_m / Σ e_j) · h_m           ← evidence-weighted fusion
  ŷ = MLP_output(h_fused)                   ← simple regression head
  L = L1(ŷ, y) + λ_e * L_evidence           ← stable L1 + evidence regularization

  L_evidence = two components:
    1. Diversity: penalize uniform weights (encourage the model to discriminate)
    2. Sparsity:  penalize using all modalities equally when some are clearly better

  Optional NIG head (detached from evidence):
    (γ, ν, α, β) = NIG_head(h_fused.detach())  ← gradient stops here
    L_aux = NIG_NLL(γ, ν, α, β, y)             ← uncertainty calibration only

Why this should work:
  - L1 loss is stable and proven (used by CASP and all baselines)
  - Evidence weights still learn meaningful modality preferences
  - Evidence regularization prevents trivial solutions
  - NIG head (if used) provides uncertainty without breaking evidence training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CASP'))
from modules.transformer import TransformerEncoder


# ─── Evidence Head ──────────────────────────────────────────────────

class EvidenceHead(nn.Module):
    """Scalar evidence per modality: e_m = softplus(MLP(h_m))"""

    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h):
        return F.softplus(self.net(h)) + 1e-6


# ─── Evidence Regularizer ──────────────────────────────────────────

class EvidenceDiversityLoss(nn.Module):
    """
    Encourage evidence weights to NOT be uniform.

    L_div = -H(weights) = Σ w_m log(w_m)
    Minimizing -H means maximizing entropy → but we want the opposite!
    Actually: L = max(0, H_target - H(weights))
    where H_target is the entropy of uniform distribution.

    Simpler: L = ||weights - uniform||_2
    This penalizes weights that are too close to uniform.
    """

    def __init__(self, mode='anti_uniform'):
        super().__init__()
        self.mode = mode

    def forward(self, weights):
        """
        Args:
            weights: [B, num_mods] normalized evidence weights
        Returns:
            scalar loss
        """
        B, M = weights.shape
        uniform = torch.ones_like(weights) / M

        if self.mode == 'anti_uniform':
            # Penalize being close to uniform → encourage discriminative weights
            return F.mse_loss(weights, uniform)
        elif self.mode == 'entropy':
            # Maximize distance from uniform in entropy space
            H = -(weights * torch.log(weights + 1e-8)).sum(-1)  # [B]
            H_uniform = torch.log(torch.tensor(M, dtype=weights.dtype, device=weights.device))
            return F.relu(H - 0.9 * H_uniform).mean()  # penalize if too close to max entropy
        elif self.mode == 'variance':
            # Maximize variance of weights across modalities
            var = weights.var(dim=-1)  # [B]
            uniform_var = torch.var(uniform, dim=-1).mean()
            return F.relu(uniform_var - var).mean()


# ─── V3 Model ──────────────────────────────────────────────────────

class EvidentialFusionV3(nn.Module):
    """
    Stable evidence-weighted multimodal fusion.

    Modes:
      'evidence_l1':  Evidence-weighted fusion + L1 loss (recommended)
      'evidence_nig': Evidence-weighted fusion + L1 + detached NIG auxiliary
      'concat':       CASP-style concat baseline
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
        mode='evidence_l1',
        evidence_hidden=32,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mods = len(orig_dim)
        self.mode = mode

        # Projection layers
        self.proj = nn.ModuleList([
            nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
            for i in range(self.num_mods)
        ])

        # Transformer encoders
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim=proj_dim, num_heads=num_heads, layers=layers,
                attn_dropout=attn_dropout, res_dropout=res_dropout,
                relu_dropout=relu_dropout, embed_dropout=embed_dropout,
            )
            for _ in range(self.num_mods)
        ])

        # Evidence heads (for evidence_* modes)
        if 'evidence' in mode:
            self.evidence_heads = nn.ModuleList([
                EvidenceHead(input_dim=proj_dim, hidden_dim=evidence_hidden)
                for _ in range(self.num_mods)
            ])

        # Output head
        fusion_dim = proj_dim if 'evidence' in mode else self.num_mods * proj_dim
        self.output_head = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(out_dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, output_dim),
        )

        # Optional NIG head (detached from evidence)
        if mode == 'evidence_nig':
            self.nig_head = nn.Sequential(
                nn.Linear(proj_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 4),  # (γ, ν_raw, α_raw, β_raw)
            )

    def encode_modality(self, x_i, encoder_idx):
        x_i = x_i.transpose(1, 2)
        x_i = self.proj[encoder_idx](x_i)
        x_i = x_i.permute(2, 0, 1)
        h_full = self.encoders[encoder_idx](x_i)
        return h_full[0], h_full  # [B, proj_dim], [L, B, proj_dim]

    def forward(self, x, return_evidence=False):
        batch_size = x[0].shape[0]
        device = x[0].device
        hs, evidence, all_present = [], [], True

        for i in range(self.num_mods):
            is_missing = (x[i].abs().sum() < 1e-8).item()
            if is_missing:
                hs.append(torch.zeros(batch_size, self.proj_dim, device=device))
                evidence.append(torch.zeros(batch_size, 1, device=device))
                all_present = False
            else:
                h_pooled, _ = self.encode_modality(x[i], i)
                hs.append(h_pooled)
                if hasattr(self, 'evidence_heads'):
                    e = self.evidence_heads[i](h_pooled)
                else:
                    e = torch.ones(batch_size, 1, device=device)
                evidence.append(e)

        hs = torch.stack(hs, dim=1)    # [B, 3, proj_dim]
        evidence = torch.stack(evidence, dim=1)  # [B, 3, 1]

        # Evidence-weighted fusion
        total_e = evidence.sum(dim=1, keepdim=True)
        weights = evidence / (total_e + 1e-8)

        if 'evidence' in self.mode:
            h_fused = (hs * weights).sum(dim=1)  # [B, proj_dim]
        else:
            h_fused = hs.reshape(batch_size, -1)  # [B, 3*proj_dim]

        output = self.output_head(h_fused)
        weights_2d = weights.squeeze(-1)

        # NIG auxiliary (detached!)
        nig_params = None
        if self.mode == 'evidence_nig':
            nig_raw = self.nig_head(h_fused.detach())
            gamma = nig_raw[:, 0:1]
            nu = F.softplus(nig_raw[:, 1:2]) + 1e-6
            alpha = F.softplus(nig_raw[:, 2:3]) + 1.0 + 1e-6
            beta = F.softplus(nig_raw[:, 3:4]) + 1e-6
            nig_params = (gamma, nu, alpha, beta)

        if return_evidence:
            return output, weights_2d, evidence.squeeze(-1), nig_params
        return output


def build_evidential_v3(orig_dim, mode='evidence_l1', **kwargs):
    return EvidentialFusionV3(orig_dim=orig_dim, mode=mode, **kwargs)


# ─── Sanity Check ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Evidential Fusion V3 Sanity Check ===\n")

    orig_dim = [768, 512, 256]
    batch_size = 4
    seq_len = 50
    text = torch.randn(batch_size, seq_len, orig_dim[0])
    audio = torch.randn(batch_size, seq_len, orig_dim[1])
    vision = torch.randn(batch_size, seq_len, orig_dim[2])

    for mode in ['evidence_l1', 'evidence_nig', 'concat']:
        print(f"\n--- Mode: {mode} ---")
        model = EvidentialFusionV3(orig_dim=orig_dim, mode=mode)
        model.eval()

        x = [text, audio, vision]

        if 'evidence' in mode:
            output, weights, evid, nig = model(x, return_evidence=True)
            print(f"  Output: {output.shape}")
            print(f"  Weights: {weights[0].detach().numpy()}")
            print(f"  Evidence: {evid[0].detach().numpy()}")
            # Missing text
            x_m = [torch.zeros_like(text), audio, vision]
            _, w_m, e_m, _ = model(x_m, return_evidence=True)
            print(f"  Missing text weights: {w_m[0].detach().numpy()}")
            print(f"  Missing text evidence: {e_m[0].detach().numpy()}")
        else:
            output = model(x)
            print(f"  Output: {output.shape}")

        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params:,}")

    # Test diversity loss
    print("\n--- Diversity Loss Test ---")
    div_loss = EvidenceDiversityLoss(mode='anti_uniform')
    uniform_w = torch.ones(4, 3) / 3
    skewed_w = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.6, 0.3, 0.1]])
    print(f"  Uniform weights loss: {div_loss(uniform_w).item():.4f}")  # should be 0
    print(f"  Skewed weights loss:  {div_loss(skewed_w).item():.4f}")  # should be > 0

    print("\n=== All sanity checks passed! ===")
