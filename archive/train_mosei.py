"""
MOSEI training with memory-efficient dataloader.
"""
import os, sys, time, argparse, random
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CASP'))
from utils.util import eval_senti
from dataloader_mosei import get_mosei_dataloader

sys.path.insert(0, os.path.dirname(__file__))
from models.evidential_uadg import EvidentialFusion, NIGLoss, EvidenceRegularizer
from models.uadg import modality_dropout


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="concat",
                   choices=["concat", "evidential"])
    p.add_argument("--datapath", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--clip", type=float, default=0.8)
    p.add_argument("--when", type=int, default=10)
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--reg_weight", type=float, default=0.1)
    p.add_argument("--name", default="mosei_model.pt")
    p.add_argument("--log_interval", type=int, default=100)
    return p.parse_args()


def train_epoch(model, optimizer, train_loader, args, epoch,
                nig_loss_fn, evid_reg_fn, l1_criterion):
    model.train()
    epoch_loss, n_batches = 0.0, len(train_loader)
    start = time.time()

    for i_batch, batch in enumerate(train_loader):
        text, audio, vision, labels = (
            batch["text"], batch["audio"], batch["vision"], batch["label"].unsqueeze(-1)
        )
        if torch.cuda.is_available():
            text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()

        model.zero_grad()

        if args.mode == 'evidential':
            output, (gamma, nu, alpha, beta) = model([text, audio, vision], return_nig=True)
            loss = nig_loss_fn(labels, gamma, nu, alpha, beta)
            loss += args.reg_weight * evid_reg_fn(labels, gamma, nu, alpha)
        else:
            output = model([text, audio, vision])
            loss = l1_criterion(output, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        epoch_loss += loss.item()

        if i_batch % args.log_interval == 0 and i_batch > 0:
            elapsed = time.time() - start
            print(f"E {epoch:2d} | B {i_batch:4d}/{n_batches:4d} | "
                  f"T {elapsed*1000/args.log_interval:5.0f}ms | Loss {loss.item():.4f}")
            start = time.time()

    return epoch_loss / n_batches


def evaluate(model, loader, args, missing=None):
    model.eval()
    l1 = nn.L1Loss()
    total_loss, results, truths = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = batch["text"], batch["audio"], batch["vision"], batch["label"].unsqueeze(-1)
            if torch.cuda.is_available():
                text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()
            if missing == 'text': text = torch.zeros_like(text)
            elif missing == 'audio': audio = torch.zeros_like(audio)
            elif missing == 'vision': vision = torch.zeros_like(vision)
            if args.mode == 'evidential':
                output, _ = model([text, audio, vision], return_nig=True)
            else:
                output = model([text, audio, vision])
            total_loss += l1(output, labels).item()
            results.append(output); truths.append(labels)
    return total_loss / len(loader), torch.cat(results), torch.cat(truths)


def main():
    args = parse_args()
    setup_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    print(f"CUDA: {use_cuda} | Mode: {args.mode} | Batch: {args.batch_size}")

    dl, orig_dim = get_mosei_dataloader(args.datapath, batch_size=args.batch_size)
    print(f"Dims: {orig_dim}")

    model = EvidentialFusion(orig_dim=orig_dim, mode=args.mode,
                             proj_dim=40, num_heads=8, layers=5)
    if use_cuda: model = model.cuda()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.when, factor=0.1)
    nig_loss_fn = NIGLoss()
    evid_reg_fn = EvidenceRegularizer()
    l1_criterion = nn.L1Loss()

    best_acc, best_epoch = 0.0, 0
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, optimizer, dl["train"], args, epoch,
                                 nig_loss_fn, evid_reg_fn, l1_criterion)
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
