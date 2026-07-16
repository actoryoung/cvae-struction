"""
Memory-efficient MOSEI dataloader.

Problem: CASP dataloader loads the 13GB pickle THREE times (once per split).
Solution: Load ONCE, share across splits, release after DataLoader creation.

Multi-process: uses file lock to serialize pickle loading so N concurrent
processes don't each load 13GB simultaneously (which OOMs the system).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import gc
import fcntl
import time
import os
import numpy as np


class MOSEIDataset(Dataset):
    """Dataset that uses pre-loaded data dict for ONE split."""

    def __init__(self, split_data, split_name, stage='pretrain',
                 selected_label=None, selected_indice=None):
        self.data = split_data
        self.split = split_name
        self.stage = stage
        # Support both list-of-arrays (legacy) and dense array formats
        txt0 = self.data["text"][0]
        if isinstance(txt0, np.ndarray):
            self.orig_dims = [txt0.shape[-1], self.data["audio"][0].shape[-1], self.data["vision"][0].shape[-1]]
        else:
            self.orig_dims = [txt0.shape[0], self.data["audio"][0].shape[0], self.data["vision"][0].shape[0]]

    def get_dim(self):
        return self.orig_dims

    def __len__(self):
        return self.data["audio"].shape[0]

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "audio": torch.tensor(self.data["audio"][idx]).float(),
            "vision": torch.tensor(self.data["vision"][idx]).float(),
            "text": torch.tensor(self.data["text"][idx]).float(),
            "label": torch.tensor(self.data["regression_labels"][idx]).float(),
        }


def get_mosei_dataloader(datapath, batch_size=8, num_workers=4, stage='pretrain',
                          selected_label=None, selected_indice=None):
    """
    Memory-efficient loader: loads pickle ONCE, creates all 3 DataLoaders,
    then releases the raw data.

    Uses a file lock to serialize pickle loading across concurrent processes,
    preventing OOM when N processes each try to load the 13GB pickle.

    Args:
        datapath: path to MOSEI pickle file
        batch_size: samples per batch
        num_workers: number of subprocesses for data loading (0=single-threaded)
        stage: 'pretrain' or 'finetune' (unused, for CASP compatibility)
    """
    # File lock to serialize pickle loading across concurrent processes
    lockdir = "/tmp/mosei_load_lock"
    os.makedirs(lockdir, exist_ok=True)
    lockfile = os.path.join(lockdir, "load.lock")

    t_wait = time.time()
    with open(lockfile, 'w') as lf:
        print(f"[Dataloader] Waiting for data lock...")
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        print(f"[Dataloader] Lock acquired (waited {time.time()-t_wait:.1f}s), loading {datapath}...")
        with open(datapath, "rb") as f:
            full_data = pickle.load(f)
        # Lock released after this block — other processes can now load
    print(f"[Dataloader] Pickle loaded, lock released.")

    # Compact: convert list-of-arrays → single dense numpy array per modality
    # The pickle stores each sample as a separate numpy array, causing massive
    # Python object overhead (13GB file → 88MB actual data). Concatenating
    # frees ~5GB per process, enabling multi-process training.
    for split in ["train", "valid", "test"]:
        sd = full_data[split]
        for key in ["text", "audio", "vision", "regression_labels"]:
            if key in sd:
                arr_list = sd[key]
                if isinstance(arr_list, list) and len(arr_list) > 0:
                    sd[key] = np.concatenate([a.reshape(1, -1) if a.ndim == 1 else a for a in arr_list], axis=0)
        # Also compact raw data lists if present
        for raw_key in ["raw_text", "raw_audio", "raw_vision"]:
            if raw_key in sd:
                del sd[raw_key]
    gc.collect()
    print(f"[Dataloader] Data compacted.")

    use_cuda = torch.cuda.is_available()
    dataloaders = {}
    for split in ["train", "valid", "test"]:
        ds = MOSEIDataset(full_data[split], split)
        shuffle = (split == "train")
        dataloaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers if split == "train" else min(2, num_workers),
            pin_memory=use_cuda,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=(num_workers > 0 and split == "train"),
        )

    orig_dim = ds.get_dim()

    # Release full_data from memory — DataLoader has references but
    # the underlying numpy arrays are shared, so this helps only partially
    del full_data
    gc.collect()

    print(f"MOSEI loaded: {len(dataloaders['train'])}/{len(dataloaders['valid'])}/{len(dataloaders['test'])} batches, dims={orig_dim}")
    return dataloaders, orig_dim
