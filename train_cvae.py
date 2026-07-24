"""Train CVAE modality reconstruction for missing modality robustness."""
import os, sys, time, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CASP'))
from utils.util import eval_senti
from dataloader_mosei import get_mosei_dataloader
from dataloader_mosei_pt import get_mosei_dataloader_pt
from models.cvae_reconstruct import CVAEMSA, kl_divergence


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="cvae", choices=["cvae", "concat", "attn_cvae", "mult", "det_mlp", "raw_mlp"])
    p.add_argument("--datapath", required=True)
    p.add_argument("--use_pt", action="store_true", help="Use preprocessed .pt files (faster, less memory)")
    p.add_argument("--dataset", default="mosei")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (0=single-thread)")
    p.add_argument("--fp16", action="store_true", help="Enable mixed-precision training (may cause NaN)")
    p.add_argument("--cvae_latent", type=int, default=32, help="CVAE latent dimension")
    p.add_argument("--cvae_hidden", type=int, default=64, help="CVAE hidden dimension")
    p.add_argument("--recon_weights", type=str, default="", help="Path to .npy per-dim reconstruction weights")
    p.add_argument("--text_dropout", type=float, default=0.0, help="T-specific dropout prob (0=uniform 1/3 per modality)")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay (L2 regularization)")
    p.add_argument("--distill_weight", type=float, default=0.0, help="Teacher→Student distillation loss weight")
    p.add_argument("--teacher_ckpt", type=str, default="checkpoints/mosei_cvae.pt", help="Frozen teacher checkpoint")
    p.add_argument("--contrastive_weight", type=float, default=0.0, help="NT-Xent cross-modal contrastive alignment weight")
    p.add_argument("--proj_dims", type=str, default="40,40,40", help="Per-modality projection dims (comma-sep, e.g. 40,64,64)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--clip", type=float, default=0.8)
    p.add_argument("--when", type=int, default=10)
    p.add_argument("--seed", type=int, default=666)
    # KL annealing
    p.add_argument("--kl_schedule", default="constant", choices=["constant","linear","cyclic"])
    p.add_argument("--kl_start", type=float, default=0.0)
    p.add_argument("--kl_end", type=float, default=0.1)
    p.add_argument("--kl_warmup", type=int, default=5)
    p.add_argument("--kl_min", type=float, default=0.01)
    p.add_argument("--kl_max", type=float, default=0.2)
    p.add_argument("--kl_period", type=int, default=10)
    # Loss weights
    p.add_argument("--kl_weight", type=float, default=0.1)
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--l2_latent", type=float, default=0.0, help="L2 reg on latent z")
    # Dropout
    p.add_argument("--dropout_prob", type=float, default=0.2)
    p.add_argument("--dropout_schedule", default="constant", choices=["constant","progressive"])
    p.add_argument("--dropout_start", type=float, default=0.05)
    p.add_argument("--dropout_end", type=float, default=0.30)
    # Cycle consistency
    p.add_argument("--cycle_weight", type=float, default=0.0, help="Cycle consistency loss weight")
    # Multi-sample inference
    p.add_argument("--mc_samples", type=int, default=1, help="MC samples at inference (1=deterministic)")
    # Subsampling for dataset-scale experiments
    p.add_argument("--subset_size", type=int, default=0, help="If >0, randomly subsample train set to N samples")
    # Output
    p.add_argument("--name", default="cvae_model.pt")
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--stage", default="pretrain")
    p.add_argument("--selected_label", default="")
    p.add_argument("--selected_indice", default="")
    return p.parse_args()


def get_kl_weight(args, epoch):
    """Compute KL weight for current epoch based on schedule."""
    if args.kl_schedule == 'constant':
        return args.kl_weight
    elif args.kl_schedule == 'linear':
        if epoch <= args.kl_warmup:
            return args.kl_start + (args.kl_end - args.kl_start) * epoch / args.kl_warmup
        return args.kl_end
    elif args.kl_schedule == 'cyclic':
        cycle_pos = (epoch % args.kl_period) / args.kl_period
        # Cosine annealing within each cycle
        return args.kl_min + 0.5 * (args.kl_max - args.kl_min) * (1 + np.cos(np.pi * cycle_pos))
    return args.kl_weight


def get_dropout_prob(args, epoch):
    """Compute dropout probability for current epoch."""
    if args.dropout_schedule == 'constant':
        return args.dropout_prob
    elif args.dropout_schedule == 'progressive':
        progress = min(epoch / args.num_epochs, 1.0)
        return args.dropout_start + (args.dropout_end - args.dropout_start) * progress
    return args.dropout_prob


def train_epoch(model, optimizer, train_loader, args, epoch, fp16=False, recon_weights=None, teacher=None, proj_contrast=None):
    # proj_contrast: nn.Linear(2*proj_dim, latent_dim) for contrastive alignment, or None
    model.train()
    l1 = nn.L1Loss()
    n_batches = len(train_loader)
    start = time.time()
    epoch_loss, epoch_reg, epoch_kl, epoch_recon = 0.0, 0.0, 0.0, 0.0
    use_amp = fp16 and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    kl_w = get_kl_weight(args, epoch)

    # Weighted dropout probs: text gets higher missing prob if --text_dropout set
    if args.text_dropout > 0:
        # text_dropout: P(text missing), other two evenly split remaining
        p_t = args.text_dropout
        p_other = (1.0 - p_t) / 2
        drop_probs = [p_t, p_other, p_other]  # [text, audio, vision]
    else:
        drop_probs = None  # uniform

    for i_batch, batch in enumerate(train_loader):
        text, audio, vision = batch["text"], batch["audio"], batch["vision"]
        labels = batch["label"].unsqueeze(-1)
        if torch.cuda.is_available():
            text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()

        model.zero_grad(set_to_none=True)
        x = [text, audio, vision]

        if args.mode in ['cvae', 'attn_cvae', 'det_mlp', 'raw_mlp']:
            is_prob = args.mode in ['cvae', 'attn_cvae']

            # Main forward (full + dropout)
            output_full, _ = model(x)
            loss_reg = l1(output_full, labels)

            drop_idx = random.choices([0, 1, 2], weights=drop_probs)[0] if drop_probs else random.randint(0, 2)
            fwd_result = model.forward_with_dropout(x, drop_idx)

            if is_prob:
                output_drop, h_recon, h_true, mu, logvar, h_avail = fwd_result
                loss_kl = kl_divergence(mu, logvar)
            else:
                output_drop, h_recon, h_true, h_avail = fwd_result
                loss_kl = 0.0
                mu, logvar = None, None  # prevent accidental use

            loss_reg_drop = l1(output_drop, labels)
            # Weighted reconstruction: prioritize predictable dims
            if recon_weights is not None:
                w = recon_weights[drop_idx].to(h_recon.device)
                loss_recon = (w * (h_recon - h_true) ** 2).mean()
            else:
                loss_recon = F.mse_loss(h_recon, h_true)
            loss_l2 = (args.l2_latent * (mu.pow(2).mean() + logvar.exp().mean())
                       if (is_prob and args.l2_latent > 0) else 0.0)

            main_loss = loss_reg + loss_reg_drop + kl_w * loss_kl + args.recon_weight * loss_recon + loss_l2

            # ── Probabilistic-only auxiliary losses ──
            if is_prob:
                # Cross-modal contrastive alignment
                if proj_contrast is not None and args.contrastive_weight > 0:
                    B = mu.shape[0]
                    z_avail = F.normalize(proj_contrast(h_avail), dim=-1)
                    z_mu = F.normalize(mu, dim=-1)
                    sim = z_avail @ z_mu.T / 0.07
                    labels_ct = torch.arange(B, device=mu.device)
                    loss_contrast = (F.cross_entropy(sim, labels_ct) + F.cross_entropy(sim.T, labels_ct)) / 2
                    main_loss = main_loss + args.contrastive_weight * loss_contrast

                # Teacher-Student distillation
                if teacher is not None and args.distill_weight > 0:
                    with torch.no_grad():
                        teacher_pred, _ = teacher(x)
                    distill_loss = F.mse_loss(output_drop, teacher_pred.detach())
                    main_loss = main_loss + args.distill_weight * distill_loss

            # Backward main loss (frees graph 1)
            if use_amp:
                scaler.scale(main_loss).backward()
            else:
                main_loss.backward()

            # Cycle consistency (probabilistic only)
            if is_prob and args.cycle_weight > 0:
                drop2 = (drop_idx + 1) % 3
                output2, h_recon2, h_true2, mu2, logvar2, _ = model.forward_with_dropout(x, drop2)
                loss_reg2 = l1(output2, labels)
                loss_kl2 = kl_divergence(mu2, logvar2)
                if recon_weights is not None:
                    w2 = recon_weights[drop2].to(h_recon2.device)
                    loss_recon2 = (w2 * (h_recon2 - h_true2) ** 2).mean()
                else:
                    loss_recon2 = F.mse_loss(h_recon2, h_true2)
                cycle_loss = args.cycle_weight * (loss_reg2 + kl_w * loss_kl2 + args.recon_weight * loss_recon2)

                if use_amp:
                    scaler.scale(cycle_loss).backward()
                else:
                    cycle_loss.backward()  # graph 2 freed immediately

                epoch_kl += (loss_kl.item() + loss_kl2.item()) / 2
                epoch_recon += (loss_recon.item() + loss_recon2.item()) / 2
            else:
                if is_prob:
                    epoch_kl += loss_kl.item()
                epoch_recon += loss_recon.item()
        else:
            output_full = model(x)[0] if isinstance(model(x), tuple) else model(x)
            main_loss = l1(output_full, labels)
            loss_reg = main_loss
            if use_amp:
                scaler.scale(main_loss).backward()
            else:
                main_loss.backward()

        # Clip + step
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        epoch_loss += main_loss.item()
        if args.mode in ['cvae', 'attn_cvae', 'det_mlp', 'raw_mlp']:
            epoch_reg += (loss_reg.item() + loss_reg_drop.item())
        else:
            epoch_reg += loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg

        if i_batch % args.log_interval == 0 and i_batch > 0:
            elapsed = time.time() - start
            msg = f"E {epoch:2d} | B {i_batch:4d}/{n_batches:4d} | T {elapsed*1000/args.log_interval:5.0f}ms | kl_w={kl_w:.3f}"
            if args.mode in ['cvae', 'attn_cvae', 'det_mlp', 'raw_mlp']:
                if args.mode in ['cvae', 'attn_cvae']:
                    msg += f" | Reg {epoch_reg/max(i_batch,1):.3f} | KL {epoch_kl/max(i_batch,1):.4f} | Recon {epoch_recon/max(i_batch,1):.4f}"
                else:
                    msg += f" | Reg {epoch_reg/max(i_batch,1):.3f} | Recon {epoch_recon/max(i_batch,1):.4f}"
            else:
                msg += f" | Loss {main_loss.item():.4f}"
            print(msg)
            start = time.time()

    return epoch_loss / n_batches


def evaluate(model, loader, args, missing=None):
    model.eval()
    l1 = nn.L1Loss()
    total_loss, results, truths = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision = batch["text"], batch["audio"], batch["vision"]
            labels = batch["label"].unsqueeze(-1)
            if torch.cuda.is_available():
                text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()
            if missing == 'text': text = torch.zeros_like(text)
            elif missing == 'audio': audio = torch.zeros_like(audio)
            elif missing == 'vision': vision = torch.zeros_like(vision)
            if args.mc_samples > 1:
                output = model.mc_forward([text, audio, vision], k=args.mc_samples)
            else:
                output, _ = model([text, audio, vision])
            total_loss += l1(output, labels).item()
            results.append(output); truths.append(labels)
    return total_loss / len(loader), torch.cat(results), torch.cat(truths)


def main():
    args = parse_args()
    setup_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    print(f"GPU: {use_cuda} | Mode: {args.mode} | KL: {args.kl_weight} | Recon: {args.recon_weight}")

    if args.dataset == 'mosei':
        if args.use_pt:
            dl, orig_dim = get_mosei_dataloader_pt(args.datapath, batch_size=args.batch_size, num_workers=args.num_workers)
        else:
            dl, orig_dim = get_mosei_dataloader(args.datapath, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CASP'))
        from utils.dataloader import getdataloader
        import types
        ns = types.SimpleNamespace(**vars(args))
        dl, orig_dim = getdataloader(ns)
    print(f"Dims: {orig_dim}")

    # Subsampling for dataset-scale experiments
    if args.subset_size > 0:
        from torch.utils.data import Subset, DataLoader
        train_ds = dl["train"].dataset
        n_train = len(train_ds)
        if args.subset_size < n_train:
            rng = torch.Generator().manual_seed(args.seed)
            indices = torch.randperm(n_train, generator=rng)[:args.subset_size].tolist()
            dl["train"] = DataLoader(
                Subset(train_ds, indices),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            print(f"Subsampled train: {n_train} → {args.subset_size} samples (seed={args.seed})")

    # Parse per-modality projection dimensions
    proj_dims = [int(x) for x in args.proj_dims.split(",")]
    if len(proj_dims) != len(orig_dim):
        print(f"Warning: proj_dims {proj_dims} length != orig_dim {len(orig_dim)}, padding")
        while len(proj_dims) < len(orig_dim):
            proj_dims.append(40)
        proj_dims = proj_dims[:len(orig_dim)]
    print(f"Proj dims: {proj_dims}")

    if args.mode == 'mult':
        from models.mult import MulT
        model = MulT(orig_dim=orig_dim, proj_dim=proj_dims[0], out_dropout=args.dropout_prob)
    elif args.mode == 'attn_cvae':
        from models.cvae_reconstruct import CVAEMSA_Attn
        model = CVAEMSA_Attn(orig_dim=orig_dim, proj_dims=proj_dims)
    elif args.mode == 'det_mlp':
        from models.cvae_reconstruct import CVAEMSA_DetMLP
        model = CVAEMSA_DetMLP(orig_dim=orig_dim, cvae_hidden=args.cvae_hidden,
                               proj_dims=proj_dims, out_dropout=args.dropout_prob)
    elif args.mode == 'raw_mlp':
        from models.cvae_reconstruct import CVAEMSA_RawMLP
        model = CVAEMSA_RawMLP(orig_dim=orig_dim, proj_dims=proj_dims,
                               out_dropout=args.dropout_prob)
    else:
        # Wire --dropout_prob to output head dropout (the main regularization knob)
        model = CVAEMSA(orig_dim=orig_dim, cvae_latent=args.cvae_latent, cvae_hidden=args.cvae_hidden,
                       proj_dims=proj_dims, out_dropout=args.dropout_prob)
    if use_cuda: model = model.cuda()
    print(f"Student params: {sum(p.numel() for p in model.parameters()):,}")

    # Load frozen teacher for distillation
    teacher = None
    if args.distill_weight > 0:
        # Teacher must use same proj_dims for compatible output
        teacher = CVAEMSA(orig_dim=orig_dim, cvae_latent=args.cvae_latent, cvae_hidden=args.cvae_hidden, proj_dims=proj_dims)
        teacher.load_state_dict(torch.load(args.teacher_ckpt, weights_only=True), strict=False)
        if use_cuda: teacher = teacher.cuda()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        print(f"Teacher loaded: {args.teacher_ckpt}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.when, factor=0.1)

    # Cross-modal contrastive alignment projection head
    proj_contrast = None
    if args.contrastive_weight > 0:
        # Input: available modalities padded to max_proj * (num_mods-1)
        max_avail = 2 * max(proj_dims)
        proj_contrast = nn.Linear(max_avail, args.cvae_latent)
        if use_cuda: proj_contrast = proj_contrast.cuda()
        optimizer.add_param_group({'params': proj_contrast.parameters()})
        print(f"Contrastive head: {max_avail} → {args.cvae_latent}")

    # Load per-modality reconstruction weights (Path 2: weighted MSE)
    recon_weights = None
    if args.recon_weights:
        import numpy as np
        w = torch.from_numpy(np.load(args.recon_weights)).float()
        if use_cuda: w = w.cuda()
        # Single weight file: apply same weights for all modalities
        recon_weights = {0: w, 1: w, 2: w}
        print(f"Loaded recon weights: {args.recon_weights}")

    best_acc, best_epoch = 0.0, 0
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, optimizer, dl["train"], args, epoch, fp16=args.fp16, recon_weights=recon_weights, teacher=teacher, proj_contrast=proj_contrast)
        val_loss, val_r, val_t = evaluate(model, dl["valid"], args)
        acc2 = eval_senti(val_r, val_t)
        scheduler.step(val_loss)
        dt = time.time() - t0
        print(f"{'─'*60}")
        print(f"Epoch {epoch:2d} | Time {dt:5.1f}s | Train {train_loss:.4f} | Val L1 {val_loss:.4f}")
        print(f"{'─'*60}")
        if acc2 > best_acc:
            best_acc, best_epoch = acc2, epoch
            torch.save(model.state_dict(), args.name)
            print(f"  >>> Best (Acc-2: {best_acc:.4f})")

    print(f"\n{'='*60}\nFINAL TEST (best ep {best_epoch}, Acc-2: {best_acc:.4f})\n{'='*60}")
    sys.stdout.flush()
    model.load_state_dict(torch.load(args.name, weights_only=True))
    for miss in [None, 'text', 'audio', 'vision']:
        _, r, t = evaluate(model, dl["test"], args, missing=miss)
        label = "Full" if miss is None else f"Missing {miss}"
        print(f"\n[{label}]")
        eval_senti(r, t)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
