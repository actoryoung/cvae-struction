"""Train Evidential Fusion V3 (Stable Hybrid)."""
import os, sys, time, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CASP'))
from utils.dataloader import getdataloader
from utils.util import eval_senti
from models.evidential_v3 import EvidentialFusionV3, EvidenceDiversityLoss, build_evidential_v3
from models.uadg import modality_dropout


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="evidence_l1", choices=["evidence_l1", "evidence_nig", "concat"])
    p.add_argument("--dataset", default="mosi")
    p.add_argument("--datapath", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--clip", type=float, default=0.8)
    p.add_argument("--when", type=int, default=10)
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--proj_dim", type=int, default=40)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=5)
    # V3 specific
    p.add_argument("--div_weight", type=float, default=0.05, help="Evidence diversity weight")
    p.add_argument("--nig_weight", type=float, default=0.01, help="NIG auxiliary loss weight")
    p.add_argument("--use_dropout", action="store_true")
    p.add_argument("--dropout_prob", type=float, default=0.15)
    # Output
    p.add_argument("--name", default="v3_model.pt")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--no_cuda", action="store_true")
    # CASP compat
    p.add_argument("--stage", default="pretrain")
    p.add_argument("--selected_label", default="")
    p.add_argument("--selected_indice", default="")
    return p.parse_args()


def nig_aux_loss(nig_params, labels):
    """Stable NIG auxiliary loss with clamping."""
    gamma, nu, alpha, beta = nig_params
    eps = 1e-6
    nu = torch.clamp(nu, max=100.0) + eps
    alpha = torch.clamp(alpha, max=100.0) + eps
    beta = beta + eps
    omega = 2.0 * beta * (1.0 + nu)
    pi_term = 0.5 * (np.log(np.pi) - torch.log(nu))
    alpha_log_omega = alpha * torch.log(omega + eps)
    error_term = (alpha + 0.5) * torch.log((labels - gamma)**2 * nu + omega + eps)
    lgamma_alpha = torch.lgamma(alpha)
    lgamma_alpha_plus_half = torch.lgamma(alpha + 0.5)
    loss = pi_term + alpha_log_omega + error_term + lgamma_alpha - lgamma_alpha_plus_half
    return torch.clamp(loss, max=100.0).mean()


def train_epoch(model, optimizer, train_loader, args, epoch, div_loss_fn):
    model.train()
    l1 = nn.L1Loss()
    epoch_loss, n_batches = 0.0, len(train_loader)
    start = time.time()

    for i_batch, batch in enumerate(train_loader):
        text, audio, vision, labels = batch["text"], batch["audio"], batch["vision"], batch["label"].unsqueeze(-1)
        if torch.cuda.is_available():
            text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()
        model.zero_grad()

        x = [text, audio, vision]
        total_loss = torch.tensor(0.0, device=labels.device)

        if args.mode in ['evidence_l1', 'evidence_nig']:
            output, weights, evidence, nig_params = model(x, return_evidence=True)
            # L1 regression (primary)
            loss_l1 = l1(output, labels)
            total_loss += loss_l1
            # Evidence diversity (prevents uniform weights)
            loss_div = div_loss_fn(weights)
            total_loss += args.div_weight * loss_div
            # NIG auxiliary (detached from evidence)
            if args.mode == 'evidence_nig' and nig_params is not None:
                loss_nig = nig_aux_loss(nig_params, labels)
                total_loss += args.nig_weight * loss_nig
            # Modality dropout
            if args.use_dropout:
                x_d, _ = modality_dropout(x, [args.dropout_prob]*3)
                output_d, _, _, _ = model(x_d, return_evidence=True)
                total_loss += l1(output_d, labels)
        else:
            output = model(x)
            total_loss = l1(output, labels)
            if args.use_dropout:
                x_d, _ = modality_dropout(x, [args.dropout_prob]*3)
                total_loss += l1(model(x_d), labels)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        epoch_loss += total_loss.item()

        if i_batch % args.log_interval == 0 and i_batch > 0:
            elapsed = time.time() - start
            print(f"Epoch {epoch:2d} | B {i_batch:3d}/{n_batches:3d} | "
                  f"T {elapsed*1000/args.log_interval:5.0f}ms | Loss {total_loss.item():.4f}")
            start = time.time()

    return epoch_loss / n_batches


def evaluate(model, loader, args, missing=None):
    model.eval()
    l1 = nn.L1Loss()
    total_loss, results, truths = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = batch["text"], batch["audio"], batch["vision"], batch["label"].unsqueeze(-1)
            if torch.cuda.is_available(): text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()
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
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(f"CUDA: {use_cuda} | Mode: {args.mode} | Div: {args.div_weight} | Dropout: {args.use_dropout}")

    dl, orig_dim = getdataloader(args)
    print(f"Data: {args.dataset} | Dims: {orig_dim}")

    model = build_evidential_v3(orig_dim, mode=args.mode, proj_dim=args.proj_dim,
                                num_heads=args.num_heads, layers=args.layers)
    if use_cuda: model = model.cuda()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.when, factor=0.1)
    div_loss_fn = EvidenceDiversityLoss(mode='anti_uniform')

    best_acc, best_epoch = 0.0, 0
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, optimizer, dl["train"], args, epoch, div_loss_fn)
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
