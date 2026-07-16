#!/usr/bin/env python3
"""Evaluate all capacity sweep checkpoints on MOSEI test set."""
import torch, os, sys, json
sys.path.insert(0, 'CASP')
from dataloader_mosei import get_mosei_dataloader
from models.cvae_reconstruct import CVAEMSA
from CASP.utils.util import eval_senti

use_cuda = torch.cuda.is_available()
dl, orig_dim = get_mosei_dataloader('casp_dataset/mosei.pkl', batch_size=8)

configs = {
    'baseline_32_64': ('checkpoints/mosei_cvae.pt', 32, 64),
    'A_64_128': ('checkpoints/mosei_cvae_A_lat64_hid128.pt', 64, 128),
    'B_64_256': ('checkpoints/mosei_cvae_B_lat64_hid256.pt', 64, 256),
    'C_128_128': ('checkpoints/mosei_cvae_C_lat128_hid128.pt', 128, 128),
    'D_128_256': ('checkpoints/mosei_cvae_D_lat128_hid256.pt', 128, 256),
}

all_results = {}

for name, (ckpt_path, latent, hidden) in configs.items():
    if not os.path.exists(ckpt_path):
        print(f"SKIP {name}: {ckpt_path} not found")
        continue

    model = CVAEMSA(orig_dim=orig_dim, cvae_latent=latent, cvae_hidden=hidden)
    if use_cuda: model = model.cuda()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    results = {}

    l1 = torch.nn.L1Loss()
    for miss, miss_name in [(None, 'Full'), ('text', 'Missing text'),
                              ('audio', 'Missing audio'), ('vision', 'Missing vision')]:
        total_loss, preds, truths = 0.0, [], []
        with torch.no_grad():
            for batch in dl['test']:
                t, a, v = batch['text'], batch['audio'], batch['vision']
                labels = batch['label'].unsqueeze(-1)
                if use_cuda: t, a, v, labels = t.cuda(), a.cuda(), v.cuda(), labels.cuda()
                if miss == 'text': t = torch.zeros_like(t)
                elif miss == 'audio': a = torch.zeros_like(a)
                elif miss == 'vision': v = torch.zeros_like(v)
                output, _ = model([t, a, v])
                total_loss += l1(output, labels).item()
                preds.append(output); truths.append(labels)

        print(f'\n[{name}] {miss_name}')
        r = torch.cat(preds); tr = torch.cat(truths)
        eval_senti(r, tr)

    print(f"  Params: {n_params:,}")

    del model
    if use_cuda: torch.cuda.empty_cache()

print("\nDone.")
