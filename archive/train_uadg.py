"""
Training script for UADG: Uncertainty-Aware Dynamic Gating
for Robust Multimodal Sentiment Analysis.

Based on CASP training pipeline, extended with:
- Modality dropout during training
- Gaussian NLL loss for uncertainty calibration
- Consistency loss between full and dropped representations

Usage:
    # Train baseline (concat mode, no uncertainty)
    python train_uadg.py --gating_mode concat --dataset mosei --datapath <path>

    # Train UADG with uncertainty gating
    python train_uadg.py --gating_mode uncertainty --dataset mosei --datapath <path>

    # Train with all improvements
    python train_uadg.py --gating_mode uncertainty --use_dropout --use_consistency \
        --dataset mosei --datapath <path>
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

# Add CASP path for dataloader and utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CASP'))
from utils.dataloader import getdataloader
from utils.util import eval_senti

# Add models path
sys.path.insert(0, os.path.dirname(__file__))
from models.uadg import (
    UADGModel, GaussianNLLLoss, ConsistencyLoss, modality_dropout
)


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="UADG Training")

    # Model
    parser.add_argument("--gating_mode", type=str, default="uncertainty",
                        choices=["uncertainty", "mlp", "uniform", "concat"],
                        help="Gating mode for fusion")
    parser.add_argument("--proj_dim", type=int, default=40)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--gate_temperature", type=float, default=1.0)
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

    # UADG-specific
    parser.add_argument("--use_dropout", action="store_true",
                        help="Use modality dropout during training")
    parser.add_argument("--dropout_prob", type=float, default=0.15,
                        help="Probability of dropping each modality")
    parser.add_argument("--use_consistency", action="store_true",
                        help="Use consistency loss between full and dropped reps")
    parser.add_argument("--consistency_weight", type=float, default=0.1)
    parser.add_argument("--use_gaussian_nll", action="store_true",
                        help="Use Gaussian NLL loss instead of L1 (requires uncertainty gating)")
    parser.add_argument("--nll_weight", type=float, default=0.1)

    # Output
    parser.add_argument("--name", type=str, default="uadg_model.pt",
                        help="Model save path")
    parser.add_argument("--log_interval", type=int, default=20)

    # CUDA
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()

    # Gaussian NLL requires uncertainty gating
    if args.use_gaussian_nll and args.gating_mode != 'uncertainty':
        print("WARNING: Gaussian NLL requires uncertainty gating. Setting --gating_mode uncertainty")
        args.gating_mode = 'uncertainty'

    return args


def train_epoch(model, optimizer, criterion, train_loader, args, epoch):
    """Train for one epoch with optional modality dropout and consistency loss."""
    model.train()
    epoch_loss = 0.0
    proc_loss = 0.0
    proc_reg_loss = 0.0
    proc_nll_loss = 0.0
    proc_cons_loss = 0.0
    proc_size = 0
    start_time = time.time()
    num_batches = len(train_loader)

    gaussian_nll = GaussianNLLLoss() if args.use_gaussian_nll else None
    consistency_loss_fn = ConsistencyLoss() if args.use_consistency else None

    for i_batch, batch in enumerate(train_loader):
        text, audio, vision, batch_Y = (
            batch["text"], batch["audio"], batch["vision"], batch["label"]
        )
        labels = batch_Y.unsqueeze(-1)  # [B, 1]

        if torch.cuda.is_available():
            text, audio, vision, labels = (
                text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()
            )

        batch_size = text.size(0)
        model.zero_grad()

        x = [text, audio, vision]
        total_loss = 0.0
        reg_loss_val = 0.0
        nll_loss_val = 0.0
        cons_loss_val = 0.0

        # Forward pass with full modalities
        if args.gating_mode == 'uncertainty':
            output, weights, sigmas_all = model(x, return_weights=True, return_sigmas=True)
            # L1 regression loss
            reg_loss = criterion(output, labels)
            reg_loss_val = reg_loss.item()
            total_loss += reg_loss

            # Gaussian NLL: mean uncertainty across modalities
            if args.use_gaussian_nll:
                mean_sigma = sigmas_all.mean(dim=-1, keepdim=True)  # [B, 1]
                nll_loss = gaussian_nll(output, mean_sigma, labels)
                nll_loss_val = nll_loss.item()
                total_loss += args.nll_weight * nll_loss
        else:
            output = model(x)
            reg_loss = criterion(output, labels)
            reg_loss_val = reg_loss.item()
            total_loss += reg_loss

        # Modality dropout + consistency
        if args.use_dropout:
            x_dropped, drop_mask = modality_dropout(
                x, drop_probs=[args.dropout_prob] * 3
            )
            if args.gating_mode == 'uncertainty':
                output_d, weights_d, sigmas_d = model(
                    x_dropped, return_weights=True, return_sigmas=True
                )
            else:
                output_d = model(x_dropped)

            # Regression loss on dropped input
            drop_reg_loss = criterion(output_d, labels)
            total_loss += drop_reg_loss

            # Consistency loss (only when uncertainty mode, using hidden states)
            if args.use_consistency and args.gating_mode == 'uncertainty':
                cons_loss = consistency_loss_fn(
                    output, output_d
                )
                cons_loss_val = cons_loss.item()
                total_loss += args.consistency_weight * cons_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        epoch_loss += total_loss.item() * batch_size
        proc_reg_loss += reg_loss_val * batch_size
        proc_nll_loss += nll_loss_val * batch_size
        proc_cons_loss += cons_loss_val * batch_size
        proc_size += batch_size

        if i_batch % args.log_interval == 0 and i_batch > 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:2d} | Batch {i_batch:3d}/{num_batches:3d} | "
                f"Time/Batch {elapsed*1000/args.log_interval:5.2f}ms | "
                f"Loss {proc_reg_loss/proc_size:.4f}"
                + (f" | NLL {proc_nll_loss/proc_size:.4f}" if args.use_gaussian_nll else "")
                + (f" | Cons {proc_cons_loss/proc_size:.4f}" if args.use_consistency else "")
            )
            proc_reg_loss, proc_nll_loss, proc_cons_loss, proc_size = 0, 0, 0, 0
            start_time = time.time()

    return epoch_loss / len(train_loader)


def evaluate(model, criterion, loader, args):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    results = []
    truths = []

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

            x = [text, audio, vision]
            output = model(x)
            total_loss += criterion(output, labels).item()

            results.append(output)
            truths.append(labels)

    avg_loss = total_loss / len(loader)
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


def evaluate_missing(model, criterion, loader, args, missing_modality):
    """Evaluate with a specific modality missing (set to zeros)."""
    model.eval()
    total_loss = 0.0
    results = []
    truths = []

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

            # Zero out the specified modality
            if missing_modality == 'text':
                text = torch.zeros_like(text)
            elif missing_modality == 'audio':
                audio = torch.zeros_like(audio)
            elif missing_modality == 'vision':
                vision = torch.zeros_like(vision)

            x = [text, audio, vision]
            output = model(x)
            total_loss += criterion(output, labels).item()

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
    print(f"CUDA available: {use_cuda}")
    print(f"Gating mode: {args.gating_mode}")
    print(f"Modality dropout: {args.use_dropout}")
    print(f"Consistency loss: {args.use_consistency}")
    print(f"Gaussian NLL: {args.use_gaussian_nll}")

    # Load data
    dataloader, orig_dim = getdataloader(args)
    train_loader = dataloader["train"]
    valid_loader = dataloader["valid"]
    test_loader = dataloader["test"]

    print(f"Data loaded: {args.dataset}")
    print(f"Orig dims: {orig_dim}")
    print(f"Train/Valid/Test: {len(train_loader)}/{len(valid_loader)}/{len(test_loader)} batches")

    # Build model
    model = UADGModel(
        orig_dim=orig_dim,
        output_dim=1,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        layers=args.layers,
        relu_dropout=args.relu_dropout,
        embed_dropout=args.embed_dropout,
        res_dropout=args.res_dropout,
        out_dropout=args.out_dropout,
        attn_dropout=args.attn_dropout,
        gate_temperature=args.gate_temperature,
        learnable_temp=True,
        gating_mode=args.gating_mode,
    )

    if use_cuda:
        model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=args.when, factor=0.1
    )

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.num_epochs + 1):
        start = time.time()

        train_loss = train_epoch(model, optimizer, criterion, train_loader, args, epoch)
        val_loss, val_results, val_truths = evaluate(model, criterion, valid_loader, args)

        acc2 = eval_senti(val_results, val_truths)
        scheduler.step(val_loss)

        duration = time.time() - start
        print("-" * 60)
        print(
            f"Epoch {epoch:2d} | Time {duration:5.1f}s | "
            f"Train Loss {train_loss:.4f} | Valid Loss {val_loss:.4f}"
        )
        print("-" * 60)

        if acc2 > best_acc:
            best_acc = acc2
            best_epoch = epoch
            torch.save(model.state_dict(), args.name)
            print(f"  -> Best model saved (Acc-2: {best_acc:.4f}) at epoch {epoch}")

    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    model.load_state_dict(torch.load(args.name))

    # 1. Full modality test
    test_loss, test_results, test_truths = evaluate(model, criterion, test_loader, args)
    print(f"\n--- Full Modality (Best Epoch {best_epoch}) ---")
    full_acc = eval_senti(test_results, test_truths)

    # 2. Missing modality tests
    for missing in ['text', 'audio', 'vision']:
        t_loss, t_results, t_truths = evaluate_missing(
            model, criterion, test_loader, args, missing
        )
        print(f"\n--- Missing {missing.capitalize()} ---")
        eval_senti(t_results, t_truths)

    # 3. Print model summary
    print(f"\nBest validation Acc-2: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {args.name}")


if __name__ == "__main__":
    main()
