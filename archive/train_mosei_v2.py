"""
Unified MOSEI training: UADG (方案A) + ContraMSA (方案B) + concat baseline.
"""
import os, sys, time, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CASP'))
from utils.util import eval_senti
from dataloader_mosei import get_mosei_dataloader

from models.uadg import UADGModel, modality_dropout as uadg_dropout
from models.contrastive_msa import ContraMSA, CrossModalContrastiveLoss, modality_dropout


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="concat",
                   choices=["concat", "uadg", "uadg_full", "contra_full", "contra_l1"])
    p.add_argument("--datapath", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--clip", type=float, default=0.8)
    p.add_argument("--when", type=int, default=10)
    p.add_argument("--seed", type=int, default=666)

    # UADG params
    p.add_argument("--gating_mode", default="uncertainty",
                   choices=["uncertainty", "mlp", "uniform", "concat"])

    # ContraMSA params
    p.add_argument("--contra_weight", type=float, default=0.2,
                   help="Cross-modal contrastive loss weight")
    p.add_argument("--contra_temp", type=float, default=0.1,
                   help="Contrastive loss temperature")

    # Shared
    p.add_argument("--use_dropout", action="store_true",
                   help="Modality dropout during training")
    p.add_argument("--dropout_prob", type=float, default=0.15)

    p.add_argument("--name", default="model.pt")
    p.add_argument("--log_interval", type=int, default=200)
    return p.parse_args()


def build_model(args, orig_dim):
    if args.model in ['uadg', 'uadg_full']:
        return UADGModel(orig_dim=orig_dim, output_dim=1, proj_dim=40,
                         num_heads=8, layers=5, gating_mode=args.gating_mode)
    elif args.model in ['contra_full', 'contra_l1']:
        mode = 'full' if args.model == 'contra_full' else 'l1'
        return ContraMSA(orig_dim=orig_dim, mode=mode, proj_dim=40, num_heads=8, layers=5)
    else:  # concat
        return UADGModel(orig_dim=orig_dim, output_dim=1, proj_dim=40,
                         num_heads=8, layers=5, gating_mode='concat')


def train_epoch(model, optimizer, train_loader, args, epoch):
    model.train()
    l1_crit = nn.L1Loss()
    contra_crit = CrossModalContrastiveLoss(temperature=args.contra_temp) if 'contra' in args.model else None
    n_batches = len(train_loader)
    start = time.time()
    epoch_loss = 0.0

    for i_batch, batch in enumerate(train_loader):
        text, audio, vision = batch["text"], batch["audio"], batch["vision"]
        labels = batch["label"].unsqueeze(-1)
        if torch.cuda.is_available():
            text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()

        model.zero_grad()
        x = [text, audio, vision]
        total = torch.tensor(0.0, device=labels.device)

        # Forward
        if args.model in ['uadg', 'uadg_full']:
            output = model(x)
            total += l1_crit(output, labels)

            if args.use_dropout and args.model == 'uadg_full':
                x_d, _ = uadg_dropout(x, [args.dropout_prob]*3)
                output_d = model(x_d)
                total += l1_crit(output_d, labels)

        elif args.model == 'contra_full':
            output, z_full = model(x, return_z=True)
            total += l1_crit(output, labels)
            # Cross-modal contrastive
            loss_contra = contra_crit(z_full, labels)
            total += args.contra_weight * loss_contra

            if args.use_dropout:
                x_d, _ = modality_dropout(x, [args.dropout_prob]*3)
                output_d, z_d = model(x_d, return_z=True)
                total += l1_crit(output_d, labels)
                # Contrastive on dropped version
                loss_contra_d = contra_crit(z_d, labels)
                total += args.contra_weight * 0.5 * loss_contra_d

        elif args.model == 'contra_l1':
            output = model(x)
            total += l1_crit(output, labels)
            if args.use_dropout:
                x_d, _ = modality_dropout(x, [args.dropout_prob]*3)
                total += l1_crit(model(x_d), labels)

        else:  # concat
            output = model(x)
            total += l1_crit(output, labels)

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        epoch_loss += total.item()

        if i_batch % args.log_interval == 0 and i_batch > 0:
            elapsed = time.time() - start
            print(f"E {epoch:2d} | B {i_batch:4d}/{n_batches:4d} | "
                  f"T {elapsed*1000/args.log_interval:5.0f}ms | Loss {total.item():.4f}")
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
            output = model([text, audio, vision])
            total_loss += l1(output, labels).item()
            results.append(output); truths.append(labels)
    return total_loss / len(loader), torch.cat(results), torch.cat(truths)


def main():
    args = parse_args()
    setup_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    print(f"GPU: {use_cuda} | Model: {args.model} | Batch: {args.batch_size}")
    if 'contra' in args.model:
        print(f"  Contra weight: {args.contra_weight}, Temp: {args.contra_temp}")

    dl, orig_dim = get_mosei_dataloader(args.datapath, batch_size=args.batch_size)
    print(f"Dims: {orig_dim}")

    model = build_model(args, orig_dim)
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
