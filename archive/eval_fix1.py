#!/usr/bin/env python3
"""Evaluate Fix 1 (T-dropout) checkpoint on MOSEI test set."""
import torch, sys
sys.path.insert(0, 'CASP')
from dataloader_mosei import get_mosei_dataloader
from models.cvae_reconstruct import CVAEMSA
from CASP.utils.util import eval_senti

use_cuda = torch.cuda.is_available()
dl, orig_dim = get_mosei_dataloader('casp_dataset/mosei.pkl', batch_size=8)
model = CVAEMSA(orig_dim=orig_dim)
if use_cuda: model = model.cuda()
model.load_state_dict(torch.load('checkpoints/mosei_cvae_fix1_tdrop.pt', weights_only=True))
model.eval()
l1 = torch.nn.L1Loss()
print(f"Best val epoch from output: grep '>>> Best' /tmp/fix1_tdrop.txt | tail -1")

for miss, name in [(None, 'Full'), ('text', 'Missing text'), ('audio', 'Missing audio'), ('vision', 'Missing vision')]:
    total_loss, results, truths = 0.0, [], []
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
            results.append(output); truths.append(labels)
    print(f'\n[{name}]')
    eval_senti(torch.cat(results), torch.cat(truths))
