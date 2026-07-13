"""Train CVAE modality reconstruction for missing modality robustness."""
import os, sys, time, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CASP'))
from utils.util import eval_senti
from dataloader_mosei import get_mosei_dataloader
from models.cvae_reconstruct import CVAEMSA, kl_divergence


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="cvae", choices=["cvae", "concat", "attn_cvae"])
    p.add_argument("--datapath", required=True)
    p.add_argument("--dataset", default="mosei")
    p.add_argument("--batch_size", type=int, default=8)
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
    # Multi-sample inference
    p.add_argument("--mc_samples", type=int, default=1, help="MC samples at inference (1=deterministic)")
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


def train_epoch(model, optimizer, train_loader, args, epoch):
    model.train()
    l1 = nn.L1Loss()
    n_batches = len(train_loader)
    start = time.time()
    epoch_loss, epoch_reg, epoch_kl, epoch_recon = 0.0, 0.0, 0.0, 0.0

    kl_w = get_kl_weight(args, epoch)

    for i_batch, batch in enumerate(train_loader):
        text, audio, vision = batch["text"], batch["audio"], batch["vision"]
        labels = batch["label"].unsqueeze(-1)
        if torch.cuda.is_available():
            text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()

        model.zero_grad()
        x = [text, audio, vision]

        if args.mode in ['cvae', 'attn_cvae']:
            output_full, _ = model(x)
            loss_reg = l1(output_full, labels)

            drop_idx = random.randint(0, 2)
            output_drop, h_recon, h_true, mu, logvar = model.forward_with_dropout(x, drop_idx)

            loss_reg += l1(output_drop, labels)
            loss_kl = kl_divergence(mu, logvar)
            loss_recon = F.mse_loss(h_recon, h_true)
            loss_l2 = args.l2_latent * (mu.pow(2).mean() + logvar.exp().mean()) if args.l2_latent > 0 else 0.0

            total = loss_reg + kl_w * loss_kl + args.recon_weight * loss_recon + loss_l2

            epoch_kl += loss_kl.item()
            epoch_recon += loss_recon.item()
        else:
            output_full = model(x)[0] if isinstance(model(x), tuple) else model(x)
            total = l1(output_full, labels)
            loss_reg = total

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        epoch_loss += total.item()
        epoch_reg += loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg

        if i_batch % args.log_interval == 0 and i_batch > 0:
            elapsed = time.time() - start
            msg = f"E {epoch:2d} | B {i_batch:4d}/{n_batches:4d} | T {elapsed*1000/args.log_interval:5.0f}ms | kl_w={kl_w:.3f}"
            if args.mode in ['cvae', 'attn_cvae']:
                msg += f" | Reg {epoch_reg/max(i_batch,1):.3f} | KL {epoch_kl/max(i_batch,1):.4f} | Recon {epoch_recon/max(i_batch,1):.4f}"
            else:
                msg += f" | Loss {total.item():.4f}"
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
        dl, orig_dim = get_mosei_dataloader(args.datapath, batch_size=args.batch_size)
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CASP'))
        from utils.dataloader import getdataloader
        import types
        ns = types.SimpleNamespace(**vars(args))
        dl, orig_dim = getdataloader(ns)
    print(f"Dims: {orig_dim}")

    if args.mode == 'attn_cvae':
        from models.cvae_reconstruct import CVAEMSA_Attn
        model = CVAEMSA_Attn(orig_dim=orig_dim)
    else:
        model = CVAEMSA(orig_dim=orig_dim)
    if use_cuda: model = model.cuda()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.when, factor=0.1)

    best_acc, best_epoch = 0.0, 0
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, optimizer, dl["train"], args, epoch)
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
    model.load_state_dict(torch.load(args.name, weights_only=True))
    for miss in [None, 'text', 'audio', 'vision']:
        _, r, t = evaluate(model, dl["test"], args, missing=miss)
        label = "Full" if miss is None else f"Missing {miss}"
        print(f"\n[{label}]")
        eval_senti(r, t)


if __name__ == "__main__":
    main()
