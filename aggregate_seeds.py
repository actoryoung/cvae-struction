#!/usr/bin/env python3
"""Aggregate multi-seed results: compute mean ± std across seeds."""
import json, glob, os
from collections import defaultdict
import numpy as np

# ─── Load all seed results ───
results = []
for pattern in ["results/seed*.json", "results/sweep/kl0.[48]*.json",
                "results/kl0.5_rw*.json", "results/mosi_*.json",
                "results/combo_kl0.[48]*.json", "results/combo_kl0.4*ct0.7*.json",
                "results/fillin_kl0.[48]*.json"]:
    for f in glob.glob(pattern):
        with open(f) as fp:
            d = json.load(fp)
        cfg = d["config"]; ft = d["final_test"]
        def g(k, v=""): return str(cfg.get(k, v))
        results.append({
            "dataset": g("dataset", "mosei"),
            "mode": g("mode", "cvae"),
            "kl": float(cfg.get("kl_weight", 0)),
            "cw": float(cfg.get("contrastive_weight", 0)),
            "seed": int(cfg.get("seed", 666)),
            "Full": ft["Full"]["Acc-2"],
            "MissT": ft["Missing text"]["Acc-2"],
            "F1_T": ft["Missing text"]["F1"],
        })

# ─── Group by (dataset, config) ───
groups = defaultdict(list)
for r in results:
    if r["mode"] == "concat":
        key = (r["dataset"], "concat")
    elif r["cw"] > 0:
        key = (r["dataset"], f"KL={r['kl']}+CT{r['cw']}")
    else:
        key = (r["dataset"], f"KL={r['kl']}")
    groups[key].append(r)

# ─── Print tables ───
for dataset in ["mosei", "mosi"]:
    print(f"\n{'='*70}")
    print(f"  {dataset.upper()}  —  Multi-Seed Results (mean ± std)")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'Seeds':>6s}  {'Full':>16s}  {'MissT':>16s}  {'F1_T':>16s}")
    print(f"{'-'*25} {'-'*6}  {'-'*16}  {'-'*16}  {'-'*16}")

    for key in sorted(groups.keys()):
        if key[0] != dataset:
            continue
        configs = groups[key]
        seeds_str = ",".join(str(r["seed"]) for r in configs)
        fl_mean = np.mean([r["Full"] for r in configs])
        fl_std = np.std([r["Full"] for r in configs])
        mt_mean = np.mean([r["MissT"] for r in configs])
        mt_std = np.std([r["MissT"] for r in configs])
        f1_mean = np.mean([r["F1_T"] for r in configs])
        f1_std = np.std([r["F1_T"] for r in configs])

        print(f"{key[1]:<25s} {seeds_str:>6s}  "
              f"{fl_mean:.4f}±{fl_std:.4f}  {mt_mean:.4f}±{mt_std:.4f}  {f1_mean:.4f}±{f1_std:.4f}")

# ─── Generate paper-ready table ───
print(f"\n{'='*70}")
print("  PAPER TABLE — MOSEI Main Results")
print(f"{'='*70}")
print(f"{'Method':<30s} {'Full Acc-2':>16s} {'MissT Acc-2':>16s}")
print(f"{'-'*30} {'-'*16} {'-'*16}")

for cfg_name in ["concat", "KL=0.4", "KL=0.8", "KL=0.4+CT0.7"]:
    key = ("mosei", cfg_name)
    if key in groups:
        g = groups[key]
        fl = f"{np.mean([r['Full'] for r in g]):.4f} ± {np.std([r['Full'] for r in g]):.4f}"
        mt = f"{np.mean([r['MissT'] for r in g]):.4f} ± {np.std([r['MissT'] for r in g]):.4f}"
        label = {"concat": "Concat (CASP)", "KL=0.4": "CVAE KL=0.4",
                 "KL=0.8": "CVAE KL=0.8", "KL=0.4+CT0.7": "CVAE KL=0.4 + CT0.7"}[cfg_name]
        print(f"{label:<30s} {fl:>16s} {mt:>16s}")
    else:
        print(f"  [Waiting for seed data: {cfg_name}]")

print(f"\nDone. Run multi_seed.sh first if any configs show as missing.")
