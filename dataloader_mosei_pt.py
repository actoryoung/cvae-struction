"""
MOSEI dataloader from preprocessed .pt files.

Loads efficiently (float32 tensors, no pickle overhead) — enables
multi-process training with 6.3 GB per process vs 13 GB for pickle.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os


class MOSEIDatasetPT(Dataset):
    """Dataset backed by preprocessed .pt file (float32 dense tensors)."""

    def __init__(self, pt_path):
        # Use mmap=True: on Linux, the OS shares the same file pages
        # across processes, so N concurrent trainers share the same 6.3GB
        # of physical memory rather than each allocating their own copy.
        self.data = torch.load(pt_path, weights_only=True, mmap=True)
        self.orig_dims = [
            self.data["text"].shape[-1],
            self.data["audio"].shape[-1],
            self.data["vision"].shape[-1],
        ]

    def get_dim(self):
        return self.orig_dims

    def __len__(self):
        return self.data["audio"].shape[0]

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "audio": self.data["audio"][idx].float(),
            "vision": self.data["vision"][idx].float(),
            "text": self.data["text"][idx].float(),
            "label": self.data["regression_labels"][idx].float(),
        }


def get_mosei_dataloader_pt(datadir, batch_size=8, num_workers=4, stage='pretrain'):
    """
    Load MOSEI from preprocessed .pt files.

    Args:
        datadir: directory containing train.pt, valid.pt, test.pt
        batch_size: samples per batch
        num_workers: DataLoader subprocesses
    """
    use_cuda = torch.cuda.is_available()
    dataloaders = {}
    for split in ["train", "valid", "test"]:
        ptpath = os.path.join(datadir, f"{split}.pt")
        print(f"Loading {ptpath}...")
        ds = MOSEIDatasetPT(ptpath)
        shuffle = (split == "train")
        dataloaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers if split == "train" else min(2, num_workers),
            pin_memory=use_cuda,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=(num_workers > 0 and split == "train"),
        )

    orig_dim = ds.get_dim()
    print(f"MOSEI loaded: {len(dataloaders['train'])}/{len(dataloaders['valid'])}/{len(dataloaders['test'])} batches, dims={orig_dim}")
    return dataloaders, orig_dim
