"""
Training script for Evidential Fusion V2.

Usage:
    # V2 evidential
    python train_evidential_v2.py --mode evidential --dataset mosi --datapath <path>

    # V2 with L1 auxiliary loss (more stable)
    python train_evidential_v2.py --mode evidential --l1_weight 0.01 --dataset mosi --datapath <path>

    # concat baseline
    python train_evidential_v2.py --mode concat --dataset mosi --datapath <path>
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CASP'))
from utils.dataloader import getdataloader
from utils.util import eval_senti

sys.path.insert(0, os.path.dirname(__file__))
from models.evidential_v2 import (
    EvidentialFusionV2, EvidentialLossV2, build_evidential_v2
)
from models.uadg import modality_dropout


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser(description="Evidential Fusion V2 Training")

    p.add_argument("--mode", default="evidential",
                   choices=["evidential", "evidential_l1", "concat"])
    p.add_argument("--proj_dim", type=int, default=40)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=5)

    p.add_argument("--dataset", default="mosi", choices=["mosi", "mosei", "sims"])
    p.add_argument("--datapath", required=True)
    p.add_argument("--stage", default="pretrain")
    p.add_argument("--selected_label", default="")
    p.add_argument("--selected_indice", default="")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--clip", type=float, default=0.8)
    p.add_argument("--when", type=int, default=10)
    p.add_argument("--seed", type=int, default=666)

    # Evidential loss
    p.add_argument("--reg_weight", type=float, default=0.1,
                   help="Evidence regularizer weight")
    p.add_argument("--l1_weight", type=float, default=0.0,
                   help="Auxiliary L1 loss weight (for stability)")

    # Modality dropout
    p.add_argument("--use_dropout", action="store_true")
    p.add_argument("--dropout_prob", type=float, default=0.15)

    # Output
    p.add_argument("--name", default="evidential_v2.pt")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--no_cuda", action="store_true")

    return p.parse_args()


def train_epoch(model, loss_fn, optimizer, train_loader, args, epoch):
    model.train()
    epoch_loss = {"nig": 0.0, "reg": 0.0, "total": 0.0}
    proc_size = 0
    start = time.time()
    n_batches = len(train_loader)

    for i_batch, batch in enumerate(train_loader):
        text, audio, vision, batch_Y = batch["text"], batch["audio"], batch["vision"], batch["label"]
        labels = batch_Y.unsqueeze(-1)

        if torch.cuda.is_available():
            text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()

        model.zero_grad()

        x = [text, audio, vision]
        total_loss = torch.tensor(0.0, device=labels.device)

        if args.mode in ['evidential', 'evidential_l1']:
            if args.mode == 'evidential':
                output, _, _, nig_params = model(x, return_evidence=True, return_nig=True)
                loss, details = loss_fn(output, nig_params, labels)
                total_loss += loss
                for k in details:
                    epoch_loss[k] += details[k] * labels.size(0)

                # Optional: modality dropout
                if args.use_dropout:
                    x_d, _ = modality_dropout(x, [args.dropout_prob] * 3)
                    output_d, _, _, nig_params_d = model(x_d, return_evidence=True, return_nig=True)
                    loss_d, _ = loss_fn(output_d, nig_params_d, labels)
                    total_loss += loss_d
            else:
                output = model(x)
                total_loss = F.l1_loss(output, labels)

        elif args.mode == 'concat':
            output = model(x)
            total_loss = F.l1_loss(output, labels)
            if args.use_dropout:
                x_d, _ = modality_dropout(x, [args.dropout_prob] * 3)
                output_d = model(x_d)
                total_loss += F.l1_loss(output_d, labels)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        proc_size += labels.size(0)

        if i_batch % args.log_interval == 0 and i_batch > 0:
            elapsed = time.time() - start
            if args.mode == 'evidential':
                print(f"Epoch {epoch:2d} | Batch {i_batch:3d}/{n_batches:3d} | "
                      f"T {elapsed*1000/args.log_interval:5.0f}ms | "
                      f"NIG {epoch_loss['nig']/proc_size:.4f} | "
                      f"Reg {epoch_loss['reg']/proc_size:.4f}")
            else:
                print(f"Epoch {epoch:2d} | Batch {i_batch:3d}/{n_batches:3d} | "
                      f"T {elapsed*1000/args.log_interval:5.0f}ms | "
                      f"L1 {epoch_loss['total']/max(proc_size,1):.4f}")
            for k in epoch_loss:
                epoch_loss[k] = 0.0
            proc_size = 0
            start = time.time()

    return epoch_loss["total"] / max(len(train_loader), 1)


def evaluate(model, loader, args, missing_modality=None):
    model.eval()
    total_loss = 0.0
    results, truths = [], []
    l1 = nn.L1Loss()

    with torch.no_grad():
        for batch in loader:
            text, audio, vision, batch_Y = batch["text"], batch["audio"], batch["vision"], batch["label"]
            labels = batch_Y.unsqueeze(-1)
            if torch.cuda.is_available():
                text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()

            if missing_modality == 'text':
                text = torch.zeros_like(text)
            elif missing_modality == 'audio':
                audio = torch.zeros_like(audio)
            elif missing_modality == 'vision':
                vision = torch.zeros_like(vision)

            if args.mode == 'evidential':
                output, _, _ = model([text, audio, vision], return_evidence=True)
            else:
                output = model([text, audio, vision])

            total_loss += l1(output, labels).item()
            results.append(output)
            truths.append(labels)

    avg_loss = total_loss / len(loader)
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


def main():
    args = parse_args()
    setup_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(f"CUDA: {use_cuda} | Mode: {args.mode} | Reg: {args.reg_weight} | L1 aux: {args.l1_weight}")

    dataloader, orig_dim = getdataloader(args)
    train_loader = dataloader["train"]
    valid_loader = dataloader["valid"]
    test_loader = dataloader["test"]
    print(f"Data: {args.dataset} | Dims: {orig_dim}")

    model = build_evidential_v2(orig_dim, mode=args.mode,
                                proj_dim=args.proj_dim, num_heads=args.num_heads,
                                layers=args.layers)
    if use_cuda:
        model = model.cuda()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.when, factor=0.1)
    loss_fn = EvidentialLossV2(reg_weight=args.reg_weight, l1_weight=args.l1_weight)

    best_acc, best_epoch = 0.0, 0

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loss_fn, optimizer, train_loader, args, epoch)
        val_loss, val_r, val_t = evaluate(model, valid_loader, args)
        acc2 = eval_senti(val_r, val_t)
        scheduler.step(val_loss)

        dt = time.time() - t0
        print(f"{'─'*60}")
        print(f"Epoch {epoch:2d} | Time {dt:5.1f}s | Val L1 {val_loss:.4f}")
        print(f"{'─'*60}")

        if acc2 > best_acc:
            best_acc, best_epoch = acc2, epoch
            torch.save(model.state_dict(), args.name)
            print(f"  >>> Best model (Acc-2: {best_acc:.4f})")

    # Final test
    print(f"\n{'='*60}\nFINAL TEST (best epoch {best_epoch}, Acc-2: {best_acc:.4f})\n{'='*60}")
    model.load_state_dict(torch.load(args.name, weights_only=True))

    for missing in [None, 'text', 'audio', 'vision']:
        _, test_r, test_t = evaluate(model, test_loader, args, missing_modality=missing)
        label = "Full Modality" if missing is None else f"Missing {missing.capitalize()}"
        print(f"\n[{label}]")
        eval_senti(test_r, test_t)


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
