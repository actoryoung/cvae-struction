#!/usr/bin/env python3
"""Aggregate multi-seed results: compute mean ± std across seeds.
Updated to handle det_mlp, raw_mlp, and subsample experiments."""
import json, glob, os
from collections import defaultdict
import numpy as np

RESULTS_DIR = "/home/ly/stu_work/projects/missing-modality-msa/results"

# ─── Load all seed results ───
results = []
patterns = [
    # Existing multi-seed results
    "results/seed*.json",
    "results/sweep/kl0.[48]*.json",
    "results/kl0.5_rw*.json",
    "results/mosi_*.json",
    "results/combo_kl0.[48]*.json",
    "results/combo_kl0.4*ct0.7*.json",
    "results/fillin_kl0.[48]*.json",
    # New experiments
    "results/det_mlp*.json",
    "results/raw_mlp*.json",
    "results/ab_recon0*.json",
    "results/sub*_concat*.json",
    "results/sub*_cvae*.json",
]
for pattern in patterns:
    for f in glob.glob(pattern):
        try:
            with open(f) as fp:
                d = json.load(fp)
            cfg = d["config"]
            ft = d["final_test"]

            def g(k, v=""): return str(cfg.get(k, v))

            results.append({
                "dataset": g("dataset", "mosei"),
                "mode": g("mode", "cvae"),
                "kl": float(cfg.get("kl_weight", 0)),
                "cw": float(cfg.get("contrastive_weight", 0)),
                "seed": int(cfg.get("seed", 666)),
                "subset_size": int(cfg.get("subset_size", 0)),
                "Full": ft["Full"]["Acc-2"],
                "MissT": ft["Missing text"]["Acc-2"],
                "MissA": ft["Missing audio"]["Acc-2"],
                "MissV": ft["Missing vision"]["Acc-2"],
                "F1_T": ft["Missing text"]["F1"],
            })
        except (KeyError, json.JSONDecodeError, FileNotFoundError):
            continue

# ─── Group by (dataset, config) ───
groups = defaultdict(list)
for r in results:
    if r["subset_size"] > 0:
        key = (r["dataset"], f"sub{r['subset_size']}_{r['mode']}_kl{r['kl']}")
    elif r["mode"] == "concat":
        key = (r["dataset"], "concat")
    elif r["mode"] == "det_mlp":
        key = (r["dataset"], "DetMLP")
    elif r["mode"] == "raw_mlp":
        key = (r["dataset"], "RawMLP")
    elif r["kl"] == 0.8 and float(cfg.get("recon_weight", 1.0)) == 0.0:
        key = (r["dataset"], "KL=0.8_Recon=0")
    elif r["cw"] > 0:
        key = (r["dataset"], f"KL={r['kl']}+CT{r['cw']}")
    else:
        key = (r["dataset"], f"KL={r['kl']}")
    groups[key].append(r)


def print_group(groups, dataset, key, label=None):
    """Print mean±std for a config group."""
    if key not in groups or len(groups[key]) < 1:
        return None
    g = groups[key]
    label = label or key[1]
    n = len(g)
    fl = np.mean([r["Full"] for r in g])
    fl_s = np.std([r["Full"] for r in g]) if n > 1 else 0
    mt = np.mean([r["MissT"] for r in g])
    mt_s = np.std([r["MissT"] for r in g]) if n > 1 else 0
    ma = np.mean([r["MissA"] for r in g])
    ma_s = np.std([r["MissA"] for r in g]) if n > 1 else 0
    mv = np.mean([r["MissV"] for r in g])
    mv_s = np.std([r["MissV"] for r in g]) if n > 1 else 0
    return label, n, fl, fl_s, mt, mt_s, ma, ma_s, mv, mv_s


# ─── Table 1: MOSEI All Methods Comparison ───
print(f"\n{'='*80}")
print("  REVIEWER EXPERIMENTS — Method Comparison (MOSEI)")
print(f"{'='*80}")
print(f"{'Method':<30s} {'N':>4s}  {'Full':>16s}  {'MissT':>16s}  {'MissA':>16s}  {'MissV':>16s}")
print(f"{'-'*30} {'-'*4}  {'-'*16}  {'-'*16}  {'-'*16}  {'-'*16}")

for key, label in [
    (("mosei", "concat"), "Concat (CASP baseline)"),
    (("mosei", "KL=0.8"), "CVAE β=0.8 (best)"),
    (("mosei", "KL=0.8_Recon=0"), "CVAE β=0.8 Recon=0 (abl)"),
    (("mosei", "DetMLP"), "Deterministic MLP (abl)"),
    (("mosei", "RawMLP"), "Raw-Feature MLP 768d"),
    (("mosei", "KL=0.4"), "CVAE β=0.4"),
    (("mosei", "KL=0.4+CT0.7"), "CVAE β=0.4+CT0.7"),
]:
    r = print_group(groups, *key, label)
    if r:
        label, n, fl, fl_s, mt, mt_s, ma, ma_s, mv, mv_s = r
        print(f"{label:<30s} {n:>4d}  {fl:.4f}±{fl_s:.4f}  {mt:.4f}±{mt_s:.4f}  "
              f"{ma:.4f}±{ma_s:.4f}  {mv:.4f}±{mv_s:.4f}")
    else:
        print(f"{label:<30s}  [NOT YET RUN]")

# ─── Table 2: Subsampling Results ───
print(f"\n{'='*80}")
print("  SUBSAMPLING — Dataset Size vs KL Preference (MOSEI)")
print(f"{'='*80}")
print(f"{'Size':>8s}  {'Concat':>16s}  {'CVAE KL=0.4':>16s}  {'CVAE KL=0.8':>16s}  {'Best KL':>10s}")
print(f"{'-'*8}  {'-'*16}  {'-'*16}  {'-'*16}  {'-'*10}")

for size in [1300, 2500, 5000, 10000, 16265]:
    concat_key = ("mosei", f"sub{size}_concat_kl0")
    cvae04_key = ("mosei", f"sub{size}_cvae_kl0.4")
    cvae08_key = ("mosei", f"sub{size}_cvae_kl0.8")

    # For full dataset (16265), use main results
    if size == 16265:
        concat_key = ("mosei", "concat")
        cvae04_key = ("mosei", "KL=0.4")
        cvae08_key = ("mosei", "KL=0.8")

    concat_r = print_group(groups, *concat_key)
    cvae04_r = print_group(groups, *cvae04_key)
    cvae08_r = print_group(groups, *cvae08_key)

    if concat_r and cvae04_r and cvae08_r:
        c_mt = concat_r[4]
        c04_mt = cvae04_r[4]
        c08_mt = cvae08_r[4]
        best = "concat" if c_mt >= max(c04_mt, c08_mt) else ("KL=0.4" if c04_mt >= c08_mt else "KL=0.8")
        print(f"{size:>8,d}  {c_mt:>16.4f}  {c04_mt:>16.4f}  {c08_mt:>16.4f}  {best:>10s}")
    else:
        missing = []
        if not concat_r: missing.append("concat")
        if not cvae04_r: missing.append("KL=0.4")
        if not cvae08_r: missing.append("KL=0.8")
        print(f"{size:>8,d}  [missing: {', '.join(missing)}]")

# ─── Key Findings ───
print(f"\n{'='*80}")
print("  KEY COMPARISONS")
print(f"{'='*80}")

# Comparison 1: CVAE vs DetMLP (VAE necessity)
for dataset in ["mosei"]:
    for cfg1, cfg2, name1, name2 in [
        (("mosei", "KL=0.8"), ("mosei", "DetMLP"), "CVAE β=0.8", "DetMLP"),
    ]:
        r1 = print_group(groups, *cfg1)
        r2 = print_group(groups, *cfg2)
        if r1 and r2:
            delta = r1[4] - r2[4]
            direction = ">" if delta > 0 else "<"
            print(f"  VAE necessity: {name1} {direction} {name2} by {delta:+.4f} MissT")
        else:
            print(f"  VAE necessity: [pending]")

# Comparison 2: Fusion-space vs Raw-space
for cfg1, cfg2, name1, name2 in [
    (("mosei", "KL=0.8"), ("mosei", "RawMLP"), "Fusion CVAE (40d)", "Raw MLP (768d)"),
    (("mosei", "DetMLP"), ("mosei", "RawMLP"), "Fusion DetMLP (40d)", "Raw MLP (768d)"),
]:
    r1 = print_group(groups, *cfg1)
    r2 = print_group(groups, *cfg2)
    if r1 and r2:
        delta = r1[4] - r2[4]
        direction = ">" if delta > 0 else "<"
        print(f"  Fusion vs Raw: {name1} {direction} {name2} by {delta:+.4f} MissT")
    else:
        print(f"  Fusion vs Raw ({name1} vs {name2}): [pending]")

print(f"\nDone. Total result files loaded: {len(results)}")
print(f"Config groups found: {len(groups)}")
