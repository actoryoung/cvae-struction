#!/usr/bin/env python3
"""Evaluate Path 2 (weighted recon) checkpoint."""
import torch, sys
sys.path.insert(0, 'CASP')
from dataloader_mosei import get_mosei_dataloader
from models.cvae_reconstruct import CVAEMSA
from CASP.utils.util import eval_senti
use_cuda = torch.cuda.is_available()
dl, orig_dim = get_mosei_dataloader('casp_dataset/mosei.pkl', batch_size=8)
model = CVAEMSA(orig_dim=orig_dim)
if use_cuda: model = model.cuda()
model.load_state_dict(torch.load('checkpoints/mosei_cvae_path2_weighted.pt', weights_only=True))
model.eval()
l1 = torch.nn.L1Loss()
for miss, name in [(None, 'Full'), ('text', 'Missing text'), ('audio', 'Missing audio'), ('vision', 'Missing vision')]:
    tl, r, t = 0.0, [], []
    with torch.no_grad():
        for batch in dl['test']:
            tx, a, v = batch['text'], batch['audio'], batch['vision']
            labels = batch['label'].unsqueeze(-1)
            if use_cuda: tx, a, v, labels = tx.cuda(), a.cuda(), v.cuda(), labels.cuda()
            if miss == 'text': tx = torch.zeros_like(tx)
            elif miss == 'audio': a = torch.zeros_like(a)
            elif miss == 'vision': v = torch.zeros_like(v)
            output, _ = model([tx, a, v])
            tl += l1(output, labels).item()
            r.append(output); t.append(labels)
    print(f'[{name}]')
    eval_senti(torch.cat(r), torch.cat(t))
