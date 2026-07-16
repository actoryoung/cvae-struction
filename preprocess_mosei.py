#!/usr/bin/env python3
"""
Preprocess MOSEI pickle → efficient .pt files.

The 13GB pickle stores each sample as a separate numpy array in a list,
causing massive Python object overhead. This script loads the pickle ONCE,
compacts each split's data into dense arrays, and saves as .pt files.

Result: ~100MB total vs 13GB pickle. Enables multi-process training.
"""
import pickle, torch, os, sys, gc, time
import numpy as np

datapath = "/home/ly/stu_work/projects/missing-modality-msa/casp_dataset/mosei.pkl"
outdir = "/home/ly/stu_work/projects/missing-modality-msa/casp_dataset/mosei_pt"
os.makedirs(outdir, exist_ok=True)

print(f"Loading {datapath} ({os.path.getsize(datapath)/1024**3:.1f} GB)...")
t0 = time.time()
with open(datapath, "rb") as f:
    full_data = pickle.load(f)
print(f"Loaded in {time.time()-t0:.0f}s. Splits: {list(full_data.keys())}")

for split in ["train", "valid", "test"]:
    sd = full_data[split]
    print(f"\n{split}:")
    tensors = {}
    total_samples = 0

    for key in ["text", "audio", "vision", "regression_labels"]:
        if key not in sd:
            continue
        arr_list = sd[key]
        if isinstance(arr_list, list):
            # Each element is a numpy array (1, D) or (L, D) → concatenate
            # Check first element shape
            first = arr_list[0]
            print(f"  {key}: list of {len(arr_list)} arrays, each shape={first.shape}, dtype={first.dtype}")
            total_samples = len(arr_list)
            # Concatenate along axis 0: (N, ...)
            dense = np.concatenate([a.reshape(1, -1) if a.ndim == 1 else a
                                     for a in arr_list], axis=0)
            tensors[key] = torch.from_numpy(dense).float()
            # Free the original list immediately
            sd[key] = None
            print(f"    → dense {dense.shape}, {dense.nbytes/1024**2:.1f} MB")
        elif isinstance(arr_list, np.ndarray):
            print(f"  {key}: ndarray shape={arr_list.shape}, dtype={arr_list.dtype}")
            tensors[key] = torch.from_numpy(arr_list).float()
            sd[key] = None

    # Save this split
    outpath = os.path.join(outdir, f"{split}.pt")
    torch.save(tensors, outpath)
    fsize = os.path.getsize(outpath) / 1024**2
    print(f"  → Saved {outpath} ({fsize:.1f} MB)")

    del tensors, sd
    gc.collect()

del full_data
gc.collect()

# Show total
total_size = sum(os.path.getsize(os.path.join(outdir, f))
                 for f in os.listdir(outdir)) / 1024**2
print(f"\nTotal: {total_size:.1f} MB (vs 13GB pickle)")
print(f"Done in {time.time()-t0:.0f}s")
