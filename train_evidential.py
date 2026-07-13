"""
Training script for Evidential Deep Learning-based Multimodal Fusion.

Implements Deep Evidential Regression (Amini et al., NeurIPS 2020)
for multimodal sentiment analysis with missing modalities.

Usage:
    # Train evidential model
    python train_evidential.py --mode evidential --dataset mosei --datapath <path>

    # Train concat baseline
    python train_evidential.py --mode concat --dataset mosei --datapath <path>

    # Train with evidence regularization
    python train_evidential.py --mode evidential --reg_weight 1.0 --dataset mosei --datapath <path>
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
from models.evidential_uadg import (
    EvidentialFusion, NIGLoss, EvidenceRegularizer, build_evidential_model
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
    parser = argparse.ArgumentParser(description="Evidential Fusion Training")

    # Model
    parser.add_argument("--mode", type=str, default="evidential",
                        choices=["evidential", "evidential_concat", "concat"],
                        help="Fusion mode")
    parser.add_argument("--proj_dim", type=int, default=40)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--relu_dropout", type=float, default=0.1)
    parser.add_argument("--embed_dropout", type=float, default=0.25)
    parser.add_argument("--res_dropout", type=float, default=0.1)
    parser.add_argument("--out_dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--dataset", type=str, default="mosei",
                        choices=["mosi", "mosei", "sims"])
    parser.add_argument("--datapath", type=str, required=True,
                        help="Path to dataset pickle file")
    parser.add_argument("--stage", type=str, default="pretrain",
                        help="Stage (required by CASP dataloader)")
    parser.add_argument("--selected_label", type=str, default="",
                        help="Pseudo-label path (only for pseudo stage)")
    parser.add_argument("--selected_indice", type=str, default="",
                        help="Pseudo-label indices (only for pseudo stage)")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--clip", type=float, default=0.8)
    parser.add_argument("--when", type=int, default=10,
                        help="LR scheduler patience")
    parser.add_argument("--seed", type=int, default=666)

    # Evidential-specific
    parser.add_argument("--reg_weight", type=float, default=0.1,
                        help="Evidence regularizer weight")
    parser.add_argument("--use_dropout", action="store_true",
                        help="Use modality dropout during training")
    parser.add_argument("--dropout_prob", type=float, default=0.15)

    # Output
    parser.add_argument("--name", type=str, default="evidential_model.pt")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--no_cuda", action="store_true")

    return parser.parse_args()


def train_epoch(model, optimizer, train_loader, args, epoch,
                nig_loss_fn, evid_reg_fn, l1_criterion):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    proc_loss, proc_reg, proc_nig = 0.0, 0.0, 0.0
    proc_size = 0
    start_time = time.time()
    num_batches = len(train_loader)

    for i_batch, batch in enumerate(train_loader):
        text, audio, vision, batch_Y = (
            batch["text"], batch["audio"], batch["vision"], batch["label"]
        )
        labels = batch_Y.unsqueeze(-1)

        if torch.cuda.is_available():
            text, audio, vision, labels = (
                text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()
            )

        batch_size = text.size(0)
        model.zero_grad()

        x = [text, audio, vision]
        total_loss = 0.0

        if args.mode == 'evidential':
            # Evidential forward: get NIG params
            output, (gamma, nu, alpha, beta) = model(x, return_nig=True)

            # NIG NLL loss
            nig_loss = nig_loss_fn(labels, gamma, nu, alpha, beta)

            # Evidence regularizer
            reg_loss = evid_reg_fn(labels, gamma, nu, alpha)

            total_loss = nig_loss + args.reg_weight * reg_loss

            proc_nig += nig_loss.item() * batch_size
            proc_reg += reg_loss.item() * batch_size

        elif args.mode == 'evidential_concat':
            output = model(x)
            total_loss = l1_criterion(output, labels)

        elif args.mode == 'concat':
            output = model(x)
            total_loss = l1_criterion(output, labels)

        # Optional modality dropout
        if args.use_dropout:
            x_dropped, drop_mask = modality_dropout(
                x, drop_probs=[args.dropout_prob] * 3
            )
            if args.mode == 'evidential':
                output_d, (gamma_d, nu_d, alpha_d, beta_d) = model(x_dropped, return_nig=True)
                drop_nig = nig_loss_fn(labels, gamma_d, nu_d, alpha_d, beta_d)
                drop_reg = evid_reg_fn(labels, gamma_d, nu_d, alpha_d)
                total_loss += drop_nig + args.reg_weight * drop_reg
            else:
                output_d = model(x_dropped)
                total_loss += l1_criterion(output_d, labels)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        epoch_loss += total_loss.item() * batch_size
        proc_size += batch_size

        if i_batch % args.log_interval == 0 and i_batch > 0:
            elapsed = time.time() - start_time
            msg = (f"Epoch {epoch:2d} | Batch {i_batch:3d}/{num_batches:3d} | "
                   f"Time/Batch {elapsed*1000/args.log_interval:5.2f}ms")
            if args.mode == 'evidential':
                msg += f" | NIG {proc_nig/proc_size:.4f} | Reg {proc_reg/proc_size:.4f}"
            else:
                msg += f" | L1 {epoch_loss/proc_size:.4f}"
            print(msg)
            proc_nig, proc_reg, proc_size = 0.0, 0.0, 0
            start_time = time.time()

    return epoch_loss / len(train_loader)


def evaluate(model, loader, args, missing_modality=None):
    """Evaluate the model, optionally with a missing modality."""
    model.eval()
    total_loss = 0.0
    results = []
    truths = []
    l1_criterion = nn.L1Loss()

    with torch.no_grad():
        for batch in loader:
            text, audio, vision, batch_Y = (
                batch["text"], batch["audio"], batch["vision"], batch["label"]
            )
            labels = batch_Y.unsqueeze(-1)

            if torch.cuda.is_available():
                text, audio, vision, labels = (
                    text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()
                )

            # Zero out specified modality
            if missing_modality == 'text':
                text = torch.zeros_like(text)
            elif missing_modality == 'audio':
                audio = torch.zeros_like(audio)
            elif missing_modality == 'vision':
                vision = torch.zeros_like(vision)

            x = [text, audio, vision]

            if args.mode == 'evidential':
                output, _ = model(x, return_nig=True)
            else:
                output = model(x)

            total_loss += l1_criterion(output, labels).item()
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
    print(f"CUDA: {use_cuda} | Mode: {args.mode} | Reg: {args.reg_weight} | Dropout: {args.use_dropout}")

    # Load data
    dataloader, orig_dim = getdataloader(args)
    train_loader = dataloader["train"]
    valid_loader = dataloader["valid"]
    test_loader = dataloader["test"]
    print(f"Data: {args.dataset} | Dims: {orig_dim} | Batches: {len(train_loader)}/{len(valid_loader)}/{len(test_loader)}")

    # Build model
    model = build_evidential_model(orig_dim, mode=args.mode,
                                   proj_dim=args.proj_dim, num_heads=args.num_heads,
                                   layers=args.layers)
    if use_cuda:
        model = model.cuda()

    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.when, factor=0.1)

    nig_loss_fn = NIGLoss()
    evid_reg_fn = EvidenceRegularizer()
    l1_criterion = nn.L1Loss()

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, optimizer, train_loader, args, epoch,
                                 nig_loss_fn, evid_reg_fn, l1_criterion)
        val_loss, val_results, val_truths = evaluate(model, valid_loader, args)
        acc2 = eval_senti(val_results, val_truths)
        scheduler.step(val_loss)

        duration = time.time() - start
        print(f"{'─'*60}")
        print(f"Epoch {epoch:2d} | Time {duration:5.1f}s | Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(f"{'─'*60}")

        if acc2 > best_acc:
            best_acc = acc2
            best_epoch = epoch
            torch.save(model.state_dict(), args.name)
            print(f"  >>> Best model (Acc-2: {best_acc:.4f})")

    # Final test
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(args.name))

    # Full modality
    _, test_r, test_t = evaluate(model, test_loader, args)
    print(f"\n[Full Modality]  (best epoch {best_epoch}, Acc-2: {best_acc:.4f})")
    eval_senti(test_r, test_t)

    # Missing modality tests
    for missing in ['text', 'audio', 'vision']:
        _, test_r, test_t = evaluate(model, test_loader, args, missing_modality=missing)
        print(f"\n[Missing {missing.capitalize()}]")
        eval_senti(test_r, test_t)


if __name__ == "__main__":
    main()
