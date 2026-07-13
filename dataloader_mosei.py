"""
Memory-efficient MOSEI dataloader.

Problem: CASP dataloader loads the 13GB pickle THREE times (once per split).
Solution: Load ONCE, share across splits, release after DataLoader creation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import gc


class MOSEIDataset(Dataset):
    """Dataset that uses pre-loaded data dict for ONE split."""

    def __init__(self, split_data, split_name, stage='pretrain',
                 selected_label=None, selected_indice=None):
        self.data = split_data
        self.split = split_name
        self.stage = stage
        self.orig_dims = [
            self.data["text"][0].shape[1],
            self.data["audio"][0].shape[1],
            self.data["vision"][0].shape[1],
        ]

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


def get_mosei_dataloader(datapath, batch_size=8, stage='pretrain',
                          selected_label=None, selected_indice=None):
    """
    Memory-efficient loader: loads pickle ONCE, creates all 3 DataLoaders,
    then releases the raw data.
    """
    print(f"Loading MOSEI from {datapath}...")
    with open(datapath, "rb") as f:
        full_data = pickle.load(f)

    dataloaders = {}
    for split in ["train", "valid", "test"]:
        ds = MOSEIDataset(full_data[split], split)
        shuffle = (split == "train")
        dataloaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    orig_dim = ds.get_dim()

    # Release full_data from memory — DataLoader has references but
    # the underlying numpy arrays are shared, so this helps only partially
    del full_data
    gc.collect()

    print(f"MOSEI loaded: {len(dataloaders['train'])}/{len(dataloaders['valid'])}/{len(dataloaders['test'])} batches, dims={orig_dim}")
    return dataloaders, orig_dim
