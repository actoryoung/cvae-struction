"""Train T2DR baseline for missing modality robustness."""
import os, sys, time, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CASP'))
from utils.util import eval_senti
from dataloader_mosei_pt import get_mosei_dataloader_pt
from models.t2dr_baseline import T2DR_MSA


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", required=True)
    p.add_argument("--dataset", default="mosei")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--clip", type=float, default=0.8)
    p.add_argument("--when", type=int, default=10)
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--dropout_prob", type=float, default=0.2)
    p.add_argument("--text_dropout", type=float, default=0.0)
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--proj_dims", type=str, default="40,40,40")
    # Output
    p.add_argument("--name", default="t2dr_model.pt")
    p.add_argument("--log_interval", type=int, default=200)
    return p.parse_args()


def train_epoch(model, optimizer, train_loader, args, epoch):
    model.train()
    l1 = nn.L1Loss()
    n_batches = len(train_loader)
    start = time.time()
    ep_total, ep_reg_full, ep_reg_drop, ep_recon = 0.0, 0.0, 0.0, 0.0

    if args.text_dropout > 0:
        p_t = args.text_dropout
        p_other = (1.0 - p_t) / 2
        drop_probs = [p_t, p_other, p_other]
    else:
        drop_probs = None

    for i_batch, batch in enumerate(train_loader):
        text, audio, vision = batch["text"], batch["audio"], batch["vision"]
        labels = batch["label"].unsqueeze(-1)
        if torch.cuda.is_available():
            text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()

        x = [text, audio, vision]
        drop_idx = random.choices([0, 1, 2], weights=drop_probs)[0] if drop_probs else random.randint(0, 2)

        optimizer.zero_grad(set_to_none=True)

        # 1. Full forward (for regression loss)
        output_full, _ = model(x)
        loss_full = l1(output_full, labels)

        # 2. Drop modality forward (for SFP reconstruction + regression)
        output_drop, h_pred, h_true, h_avail = model.forward_with_dropout(x, drop_idx)
        loss_drop = l1(output_drop, labels)
        loss_recon = F.mse_loss(h_pred, h_true)

        total = loss_full + loss_drop + args.recon_weight * loss_recon
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        ep_total += total.item()
        ep_reg_full += loss_full.item()
        ep_reg_drop += loss_drop.item()
        ep_recon += loss_recon.item()

        if i_batch % args.log_interval == 0 and i_batch > 0:
            elapsed = time.time() - start
            msg = (f"E {epoch:2d} | B {i_batch:4d}/{n_batches:4d} | "
                   f"T {elapsed*1000/args.log_interval:5.0f}ms | "
                   f"RegF {ep_reg_full/max(i_batch,1):.3f} | "
                   f"RegD {ep_reg_drop/max(i_batch,1):.3f} | "
                   f"Recon {ep_recon/max(i_batch,1):.4f}")
            print(msg)
            start = time.time()

    return ep_total / n_batches


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
    print(f"GPU: {use_cuda} | T2DR | recon={args.recon_weight}")

    dl, orig_dim = get_mosei_dataloader_pt(
        args.datapath, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"Dims: {orig_dim}")

    proj_dims = [int(x) for x in args.proj_dims.split(",")]
    if len(proj_dims) != len(orig_dim):
        while len(proj_dims) < len(orig_dim):
            proj_dims.append(40)
        proj_dims = proj_dims[:len(orig_dim)]

    model = T2DR_MSA(orig_dim=orig_dim, proj_dims=proj_dims, out_dropout=args.dropout_prob)
    if use_cuda: model = model.cuda()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
