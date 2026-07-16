"""
CVAE Modality Reconstruction for Missing Modality Robustness.

When a modality is missing at test time, instead of just zero-filling it
(which all previous approaches did), we use a lightweight Conditional VAE
to RECONSTRUCT the missing modality's latent representation from the
available modalities.

Key design:
  - Reconstruction happens in proj_dim=40 fusion space (NOT raw 768/512/171)
  - CVAE is very lightweight (~50K params)
  - Supports any missing modality combination
  - Trained jointly with the main regression task

Training:
  Full: [text, audio, vision] → Encoders → [h_t, h_a, h_v] → Concat → Prediction
  Mask: randomly zero one modality
  CVAE: available h's → encode → sample z → decode → reconstructed h_missing
  Loss: L1(pred, y) + KL(q(z|h_avail, h_missing) || p(z|h_avail)) + MSE(recon, true)

Inference (text missing):
  [audio, vision] → Encoders → [h_a, h_v]
  CVAE: [h_a, h_v] → encode → sample z → decode → h_t_reconstructed
  Concat[h_t_reconstructed, h_a, h_v] → Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CASP'))
from modules.transformer import TransformerEncoder


# ─── Conditional VAE for Modality Reconstruction ────────────────────

class CVAEModalityReconstructor(nn.Module):
    """
    Conditional VAE: reconstructs a missing modality's latent representation
    from available modalities.

    Encoder: q(z | h_available, h_missing_true)  [only used during training]
    Decoder: p(h_missing | z, h_available)        [used during both train and test]
    Prior:   p(z | h_available)                    [standard Gaussian]

    Architecture:
      h_avail: concat of available h_m's [B, k*proj_dim]
      Encoder: concat(h_avail, h_missing) → Linear → [μ, logvar]
      Decoder: concat(z, h_avail) → MLP → reconstructed h_missing
    """

    def __init__(self, proj_dim=40, num_mods=3, latent_dim=32, hidden_dim=64, proj_dims=None):
        super().__init__()
        # Support both single proj_dim (backward compat) and per-modality proj_dims
        if proj_dims is None:
            proj_dims = [proj_dim] * num_mods
        self.proj_dims = proj_dims
        self.max_proj = max(proj_dims)
        self.num_mods = num_mods
        self.latent_dim = latent_dim
        self.proj_dim = self.max_proj  # for backward compat

        # Max encoder input: (num_mods-1)*max_proj (padded avail) + max_proj (padded target)
        max_enc_input = (num_mods - 1) * self.max_proj + self.max_proj
        # Max decoder input: latent + (num_mods-1)*max_proj (padded avail)
        max_dec_input = latent_dim + (num_mods - 1) * self.max_proj

        self.encoder = nn.Sequential(
            nn.Linear(max_enc_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(max_dec_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.max_proj),  # always output max_proj
        )

    def _pad_features(self, hs):
        """Pad each h to max_proj dim, then concat."""
        padded = []
        for h in hs:
            d = h.shape[-1]
            if d < self.max_proj:
                zeros = torch.zeros(*h.shape[:-1], self.max_proj - d, device=h.device, dtype=h.dtype)
                h = torch.cat([h, zeros], dim=-1)
            padded.append(h)
        return torch.cat(padded, dim=-1)

    def encode(self, h_avail, h_missing_true):
        """q(z | h_avail, h_missing_true) → μ, logvar"""
        x = torch.cat([h_avail, h_missing_true], dim=-1)
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z ~ N(mu, sigma) using reparameterization trick (exp in fp32 for stability)."""
        if self.training:
            std = torch.exp(0.5 * logvar.float()).to(logvar.dtype)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, h_avail):
        """p(h_missing | z, h_avail) → reconstructed h_missing [B, max_proj]"""
        x = torch.cat([z, h_avail], dim=-1)
        return self.decoder(x)

    def forward(self, h_avail_list, h_missing_true, drop_idx):
        """
        Args:
            h_avail_list: list of available h tensors [each [B, proj_dims[i]]]
            h_missing_true: [B, target_dim] true missing modality h
            drop_idx: which modality is missing (0=T, 1=A, 2=V)
        Returns:
            h_recon: [B, target_dim] — sliced to correct dimension
            (mu, logvar): if training
        """
        target_dim = self.proj_dims[drop_idx]
        h_avail_padded = self._pad_features(h_avail_list)

        # Pad target to max_proj for encoder
        if h_missing_true.shape[-1] < self.max_proj:
            pad = torch.zeros(h_missing_true.shape[0], self.max_proj - h_missing_true.shape[-1],
                              device=h_missing_true.device, dtype=h_missing_true.dtype)
            h_missing_padded = torch.cat([h_missing_true, pad], dim=-1)
        else:
            h_missing_padded = h_missing_true

        if self.training:
            mu, logvar = self.encode(h_avail_padded, h_missing_padded)
            z = self.reparameterize(mu, logvar)
            h_recon_full = self.decode(z, h_avail_padded)
            return h_recon_full[:, :target_dim], mu, logvar
        else:
            batch_size = h_avail_padded.shape[0]
            device = h_avail_padded.device
            z = torch.zeros(batch_size, self.latent_dim, device=device)
            h_recon_full = self.decode(z, h_avail_padded)
            return h_recon_full[:, :target_dim]

    def reconstruct(self, h_avail_list, drop_idx):
        """Reconstruct missing modality. z~N(0,I) for MC robustness."""
        target_dim = self.proj_dims[drop_idx]
        h_avail_padded = self._pad_features(h_avail_list)
        batch_size = h_avail_padded.shape[0]
        device = h_avail_padded.device
        z = torch.randn(batch_size, self.latent_dim, device=device)
        h_recon_full = self.decode(z, h_avail_padded)
        return h_recon_full[:, :target_dim]


def kl_divergence(mu, logvar):
    """KL(N(mu, sigma) || N(0, I)) — computed in fp32 for numerical stability."""
    mu32, lv32 = mu.float(), logvar.float()
    return (-0.5 * torch.sum(1 + lv32 - mu32.pow(2) - lv32.exp(), dim=-1).mean())


# ─── CVAE-Enhanced MSA Model ────────────────────────────────────────

class CVAEMSA(nn.Module):
    """
    MSA model with CVAE modality reconstruction for missing modality robustness.

    Supports:
      - Any missing modality (text, audio, or vision)
      - Joint training of CVAE + regression
      - Modality dropout during training
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
        cvae_latent=32,
        cvae_hidden=64,
        proj_dims=None,
    ):
        super().__init__()
        self.orig_dim = orig_dim
        self.num_mods = len(orig_dim)
        # Support per-modality projection dimensions
        if proj_dims is None:
            proj_dims = [proj_dim] * self.num_mods
        self.proj_dims = proj_dims
        self.proj_dim = sum(proj_dims)  # total concat dim (for backward compat)

        # Per-modality projections (each with its own dim)
        self.proj = nn.ModuleList([
            nn.Conv1d(self.orig_dim[i], self.proj_dims[i], kernel_size=1, padding=0)
            for i in range(self.num_mods)
        ])
        # Per-modality Transformer encoders (each with its own embed_dim)
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim=self.proj_dims[i], num_heads=num_heads, layers=layers,
                attn_dropout=attn_dropout, res_dropout=res_dropout,
                relu_dropout=relu_dropout, embed_dropout=embed_dropout,
            )
            for i in range(self.num_mods)
        ])

        # CVAE for modality reconstruction (handles asymmetric dims via padding)
        self.cvae = CVAEModalityReconstructor(
            proj_dim=max(proj_dims), num_mods=self.num_mods,
            latent_dim=cvae_latent, hidden_dim=cvae_hidden,
            proj_dims=proj_dims,
        )

        # Output head: sum(proj_dims) → output_dim (compressing pyramid)
        total_dim = sum(proj_dims)
        self.output_head = nn.Sequential(
            nn.Linear(total_dim, total_dim * 2 // 3),
            nn.ReLU(),
            nn.Dropout(out_dropout),
            nn.Linear(total_dim * 2 // 3, total_dim // 3),
            nn.ReLU(),
            nn.Linear(total_dim // 3, output_dim),
        )

    def encode_modality(self, x_i, encoder_idx):
        x_i = x_i.transpose(1, 2)
        x_i = self.proj[encoder_idx](x_i)
        x_i = x_i.permute(2, 0, 1)
        h_full = self.encoders[encoder_idx](x_i)
        return h_full[0], h_full

    def forward(self, x, missing_idx=None, return_cvae_loss=False):
        """
        Args:
            x: [text, audio, vision] tensors
            missing_idx: if set, which modality to treat as missing (0/1/2)
            return_cvae_loss: if True, also return (mu, logvar, h_true) for CVAE loss
        Returns:
            output: [B, 1]
            (optional) cvae_data: (h_recon, mu, logvar, h_missing_true) for KL+MSE loss
        """
        batch_size = x[0].shape[0]
        device = x[0].device
        hs = []
        is_missing = []

        for i in range(self.num_mods):
            miss = (x[i].abs().sum() < 1e-8).item() or (missing_idx is not None and i == missing_idx)
            is_missing.append(miss)
            if miss:
                hs.append(None)  # placeholder, will be filled by CVAE
            else:
                h_pooled, _ = self.encode_modality(x[i], i)
                hs.append(h_pooled)

        # CVAE reconstruction for missing modalities
        cvae_data = None
        available_hs = [hs[i] for i in range(self.num_mods) if not is_missing[i]]
        missing_indices = [i for i in range(self.num_mods) if is_missing[i]]

        if missing_indices and len(available_hs) > 0:
            for mi in missing_indices:
                h_recon = self.cvae.reconstruct(available_hs, mi)
                hs[mi] = h_recon

        # Fill any remaining None with zeros of correct dimension
        for i in range(self.num_mods):
            if hs[i] is None:
                hs[i] = torch.zeros(batch_size, self.proj_dims[i], device=device)

        # Concat all and predict
        h_cat = torch.cat(hs, dim=-1)
        output = self.output_head(h_cat)

        return output, None

    def mc_forward(self, x, k=5):
        """Multi-sample inference: average predictions over K latent samples.
        Uses sampled z ~ N(0,I) instead of z=0 for missing modalities."""
        self.eval()
        batch_size = x[0].shape[0]
        device = x[0].device
        hs, is_missing = [], []
        for i in range(self.num_mods):
            miss = (x[i].abs().sum() < 1e-8).item()
            is_missing.append(miss)
            if miss:
                hs.append(None)
            else:
                h_pooled, _ = self.encode_modality(x[i], i)
                hs.append(h_pooled)
        available_hs = [hs[i] for i in range(self.num_mods) if not is_missing[i] and hs[i] is not None]
        missing_indices = [i for i in range(self.num_mods) if is_missing[i]]
        if missing_indices and len(available_hs) > 0:
            h_avail_padded = self.cvae._pad_features(available_hs)
            outputs = []
            for _ in range(k):
                hs_k = hs.copy()
                for mi in missing_indices:
                    z = torch.randn(batch_size, self.cvae.latent_dim, device=device)
                    h_recon_full = self.cvae.decode(z, h_avail_padded)
                    target_dim = self.proj_dims[mi]
                    hs_k[mi] = h_recon_full[:, :target_dim]
                for i in range(self.num_mods):
                    if hs_k[i] is None:
                        hs_k[i] = torch.zeros(batch_size, self.proj_dims[i], device=device)
                h_cat = torch.cat(hs_k, dim=-1)
                outputs.append(self.output_head(h_cat))
            output = torch.stack(outputs).mean(dim=0)
        else:
            for i in range(self.num_mods):
                if hs[i] is None:
                    hs[i] = torch.zeros(batch_size, self.proj_dims[i], device=device)
            h_cat = torch.cat(hs, dim=-1)
            output = self.output_head(h_cat)
        return output

    def forward_with_dropout(self, x, drop_idx):
        """
        Forward with a specific modality dropped for CVAE training.

        Args:
            x: full modality input
            drop_idx: which modality to drop (0=text, 1=audio, 2=vision)
        Returns:
            output, h_recon, h_missing_true, mu, logvar, h_avail_padded
        """
        # Encode all modalities
        hs_true = []
        for i in range(self.num_mods):
            h, _ = self.encode_modality(x[i], i)
            hs_true.append(h)

        available_hs = [hs_true[i] for i in range(self.num_mods) if i != drop_idx]
        h_missing_true = hs_true[drop_idx]

        # CVAE asymmetric forward: pass list + drop_idx
        h_recon, mu, logvar = self.cvae(available_hs, h_missing_true, drop_idx)

        # h_avail padded (for contrastive alignment)
        h_avail_padded = self.cvae._pad_features(available_hs)

        # Build full h list
        hs_final = []
        for i in range(self.num_mods):
            if i == drop_idx:
                hs_final.append(h_recon)
            else:
                hs_final.append(hs_true[i])

        h_cat = torch.cat(hs_final, dim=-1)
        output = self.output_head(h_cat)

        return output, h_recon, h_missing_true, mu, logvar, h_avail_padded


def modality_dropout(x, drop_probs=None):
    """Randomly drop modalities. At least one stays."""
    if drop_probs is None:
        drop_probs = [0.15, 0.15, 0.15]
    x_d, mask = [], []
    dropped = []
    for i, (xi, p) in enumerate(zip(x, drop_probs)):
        if torch.rand(1).item() < p:
            x_d.append(torch.zeros_like(xi))
            mask.append(0.0)
            dropped.append(i)
        else:
            x_d.append(xi)
            mask.append(1.0)
    if sum(mask) < 0.5:
        keep = torch.randint(0, len(x), (1,)).item()
        x_d[keep] = x[keep]
        mask[keep] = 1.0
        dropped = [i for i in range(len(x)) if i != keep]
    return x_d, mask, dropped


# ─── Gumbel-Softmax Module ──────────────────────────────────────────

class GumbelGate(nn.Module):
    """
    Gumbel-Softmax based soft gating for missing modalities.

    During training: uses Gumbel-Softmax with temperature annealing
    During inference: uses hard argmax
    """

    def __init__(self, input_dim, num_mods=3, initial_temp=1.0, min_temp=0.1):
        super().__init__()
        self.num_mods = num_mods
        self.temp = initial_temp
        self.min_temp = min_temp

        # MLP to predict logits from fused representation
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_mods),
        )

    def forward(self, h_fused):
        """
        Args:
            h_fused: [B, proj_dim*3] fused representation
        Returns:
            weights: [B, num_mods] soft weights (Gumbel-softmax in training)
        """
        logits = self.net(h_fused)

        if self.training:
            weights = F.gumbel_softmax(logits, tau=self.temp, hard=False, dim=-1)
        else:
            weights = F.softmax(logits, dim=-1)

        return weights

    def step_temp(self, decay=0.99):
        """Anneal temperature toward min_temp."""
        self.temp = max(self.min_temp, self.temp * decay)


# ─── Attention-Guided CVAE ─────────────────────────────────────────
class AttnCVAEReconstructor(nn.Module):
    """CVAE with attention-based decoder that selectively attends to available modalities."""
    def __init__(self, proj_dim=40, num_mods=3, latent_dim=32, hidden_dim=64, num_heads=4):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_mods = num_mods
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        max_input = num_mods * proj_dim
        self.encoder = nn.Sequential(
            nn.Linear(max_input, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Query projection from latent z
        self.z_proj = nn.Linear(latent_dim, proj_dim)
        # Cross-attention: z attends to available modality features
        self.cross_attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True)
        # Decoder after attention
        self.decoder = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def encode(self, h_avail, h_missing_true):
        x = torch.cat([h_avail, h_missing_true], dim=-1)
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar.float()).to(logvar.dtype)
            return mu + torch.randn_like(mu) * std
        return mu

    def decode(self, z, h_avail, h_list):
        """Attention-guided decode: z attends to individual modality features."""
        B = z.shape[0]
        # Stack available modality features as keys/values: [B, num_avail, proj_dim]
        h_stack = torch.stack(h_list, dim=1)  # [B, k, proj_dim]
        # Query from z: [B, 1, proj_dim]
        query = self.z_proj(z).unsqueeze(1)
        # Cross-attention
        attn_out, _ = self.cross_attn(query, h_stack, h_stack)
        # Decode: concat(z_proj, attn_out)
        decoder_input = torch.cat([query.squeeze(1), attn_out.squeeze(1)], dim=-1)
        return self.decoder(decoder_input)

    def forward(self, h_avail, h_list_avail, h_missing_true=None):
        if self.training and h_missing_true is not None:
            mu, logvar = self.encode(h_avail, h_missing_true)
            z = self.reparameterize(mu, logvar)
            h_recon = self.decode(z, h_avail, h_list_avail)
            return h_recon, mu, logvar
        else:
            B, D = h_avail.shape[0], self.latent_dim
            z = torch.zeros(B, D, device=h_avail.device)
            h_recon = self.decode(z, h_avail, h_list_avail)
            return h_recon

    def reconstruct(self, h_avail, h_list_avail=None):
        """Reconstruct with random z for MC robustness."""
        B, D = h_avail.shape[0], self.latent_dim
        z = torch.randn(B, D, device=h_avail.device)
        if h_list_avail:
            return self.decode(z, h_avail, h_list_avail)
        else:
            return self.decode(z, h_avail)


class CVAEMSA_Attn(CVAEMSA):
    """CVAE-MSA with attention-guided decoder."""
    def __init__(self, *args, **kwargs):
        kwargs.pop('cvae_latent', None)
        kwargs.pop('cvae_hidden', None)
        super().__init__(*args, **kwargs)
        self.cvae = AttnCVAEReconstructor(proj_dim=self.proj_dim, num_mods=self.num_mods,
                                           latent_dim=32, hidden_dim=64)

    def forward(self, x, return_cvae_loss=False):
        """Override to pass per-modality h_list to attention CVAE."""
        batch_size = x[0].shape[0]
        device = x[0].device
        hs, is_missing = [], []
        for i in range(self.num_mods):
            miss = (x[i].abs().sum() < 1e-8).item()
            is_missing.append(miss)
            if miss:
                hs.append(None)
            else:
                h_pooled, _ = self.encode_modality(x[i], i)
                hs.append(h_pooled)

        available_hs = [hs[i] for i in range(self.num_mods) if not is_missing[i] and hs[i] is not None]
        missing_indices = [i for i in range(self.num_mods) if is_missing[i]]

        if missing_indices and len(available_hs) > 0:
            h_avail = torch.cat(available_hs, dim=-1)
            for mi in missing_indices:
                h_recon = self.cvae.reconstruct(h_avail, available_hs)
                hs[mi] = h_recon

        for i in range(self.num_mods):
            if hs[i] is None:
                hs[i] = torch.zeros(batch_size, self.proj_dim, device=device)

        h_cat = torch.cat(hs, dim=-1)
        output = self.output_head(h_cat)
        all_hs = torch.stack(hs, dim=1)
        return output, all_hs

    def forward_with_dropout(self, x, drop_idx):
        batch_size = x[0].shape[0]
        device = x[0].device
        hs_true = []
        for i in range(self.num_mods):
            h, _ = self.encode_modality(x[i], i)
            hs_true.append(h)

        available_hs = [hs_true[i] for i in range(self.num_mods) if i != drop_idx]
        h_avail = torch.cat(available_hs, dim=-1)
        h_missing_true = hs_true[drop_idx]

        h_recon, mu, logvar = self.cvae(h_avail, available_hs, h_missing_true)

        hs_final = []
        for i in range(self.num_mods):
            hs_final.append(h_recon if i == drop_idx else hs_true[i])

        h_cat = torch.cat(hs_final, dim=-1)
        output = self.output_head(h_cat)
        return output, h_recon, h_missing_true, mu, logvar, h_avail


# ─── Sanity Check ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== CVAE Modality Reconstruction Sanity Check ===\n")

    orig_dim = [768, 512, 256]
    B, L = 4, 50
    text = torch.randn(B, L, orig_dim[0])
    audio = torch.randn(B, L, orig_dim[1])
    vision = torch.randn(B, L, orig_dim[2])

    # Test 1: CVAE module standalone
    print("--- CVAE Module ---")
    cvae = CVAEModalityReconstructor(proj_dim=40, latent_dim=32, hidden_dim=64)
    cvae.train()

    # Sim: text missing, audio+vision available
    h_a = torch.randn(B, 40)
    h_v = torch.randn(B, 40)
    h_t = torch.randn(B, 40)  # "true" text representation
    h_avail = torch.cat([h_a, h_v], dim=-1)  # [B, 80]

    h_recon, mu, logvar = cvae(h_avail, h_t)
    kl = kl_divergence(mu, logvar)
    mse = F.mse_loss(h_recon, h_t)
    print(f"  Training: h_recon {h_recon.shape}, KL {kl:.4f}, MSE {mse:.4f}")

    cvae.eval()
    h_recon_inf = cvae.reconstruct(h_avail)
    print(f"  Inference: h_recon {h_recon_inf.shape}")

    # Test 2: Full CVAE-MSA model
    print("\n--- CVAE-MSA Model ---")
    model = CVAEMSA(orig_dim=orig_dim)
    model.train()

    # Full modality forward
    x = [text, audio, vision]
    output, all_hs = model(x)
    print(f"  Full: output {output.shape}, hs {all_hs.shape}")

    # Drop text during training
    output_d, h_recon, h_true, mu, logvar = model.forward_with_dropout(x, drop_idx=0)
    kl = kl_divergence(mu, logvar)
    mse = F.mse_loss(h_recon, h_true)
    print(f"  Drop text: output {output_d.shape}, KL {kl:.4f}, MSE {mse:.4f}")

    # Missing text at inference
    model.eval()
    x_miss = [torch.zeros_like(text), audio, vision]
    output_m, all_hs_m = model(x_miss)
    print(f"  Missing text (inference): output {output_m.shape}")

    # Test 3: Gumbel Gate
    print("\n--- Gumbel Gate ---")
    gate = GumbelGate(input_dim=40*3, num_mods=3)
    gate.train()
    w = gate(torch.randn(B, 120))
    print(f"  Training weights: {w[0].detach().numpy()}")
    gate.eval()
    w_hard = gate(torch.randn(B, 120))
    print(f"  Inference weights: {w_hard[0].detach().numpy()}")

    # Params
    total = sum(p.numel() for p in model.parameters())
    cvae_params = sum(p.numel() for p in cvae.parameters())
    print(f"\n  CVAE params: {cvae_params:,}")
    print(f"  Total params: {total:,}")

    print("\n=== All sanity checks passed! ===")
