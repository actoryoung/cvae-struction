#!/usr/bin/env python3
"""
Aggregate all grid sweep results and produce:
  1. Ranked table sorted by MissT Acc-2
  2. Top-10 configurations
  3. Best per hyperparameter value (marginal analysis)
"""
import json, os, sys
from collections import defaultdict

RESULTS_DIR = "/home/ly/stu_work/projects/missing-modality-msa/results/sweep"

def load_all():
    records = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(RESULTS_DIR, fname)
        try:
            with open(path) as f:
                r = json.load(f)
            ft = r.get('final_test', {})
            records.append({
                'name': r['task'],
                'config': r.get('config', {}),
                'best_epoch': r.get('best_epoch'),
                'best_val': r.get('best_val_acc2'),
                'full': ft.get('Full', {}).get('Acc-2'),
                'miss_t': ft.get('Missing text', {}).get('Acc-2'),
                'miss_a': ft.get('Missing audio', {}).get('Acc-2'),
                'miss_v': ft.get('Missing vision', {}).get('Acc-2'),
                'full_f1': ft.get('Full', {}).get('F1'),
                'miss_t_f1': ft.get('Missing text', {}).get('F1'),
            })
        except Exception as e:
            print(f"Error loading {fname}: {e}", file=sys.stderr)
    return records

def main():
    records = load_all()
    if not records:
        print("No results found. Sweep may still be running.")
        return

    # Sort by MissT (descending — higher is better)
    records.sort(key=lambda r: r['miss_t'] or 0, reverse=True)

    print(f"{'='*80}")
    print(f"Grid Sweep Summary: {len(records)} configurations evaluated")
    print(f"{'='*80}\n")

    # ── Top 10 ──
    print("🏆 Top 10 by Missing Text Acc-2:")
    print(f"{'Rank':<5} {'Config':<45} {'Full':>7} {'MissT':>7} {'MissA':>7} {'MissV':>7} {'BestEp':>7}")
    print("-" * 95)
    for i, r in enumerate(records[:10]):
        cfg_short = f"kl={r['config'].get('kl_weight','?')} rw={r['config'].get('recon_weight','?')} dp={r['config'].get('dropout_prob','?')} lr={r['config'].get('lr','?')}"
        print(f"{i+1:<5} {cfg_short:<45} {r['full'] or 0:7.4f} {r['miss_t'] or 0:7.4f} {r['miss_a'] or 0:7.4f} {r['miss_v'] or 0:7.4f} {r['best_epoch'] or 0:7d}")

    # ── Full ranked list ──
    print(f"\n\n{'='*80}")
    print("Full Ranked List:")
    print(f"{'='*80}")
    for i, r in enumerate(records):
        cfg_short = f"kl={r['config'].get('kl_weight','?')} rw={r['config'].get('recon_weight','?')} dp={r['config'].get('dropout_prob','?')} lr={r['config'].get('lr','?')}"
        print(f"{i+1:3d}. {cfg_short:<45} MissT={r['miss_t'] or 0:.4f}  Full={r['full'] or 0:.4f}")

    # ── Marginal analysis: best MissT per hyperparameter value ──
    print(f"\n\n{'='*80}")
    print("Marginal Analysis: Best Average MissT per Hyperparameter Value")
    print(f"{'='*80}")

    for param in ['kl_weight', 'recon_weight', 'dropout_prob', 'lr']:
        groups = defaultdict(list)
        for r in records:
            val = r['config'].get(param, '?')
            if r['miss_t'] is not None:
                groups[val].append(r['miss_t'])
        print(f"\n  {param}:")
        for val in sorted(groups.keys()):
            scores = groups[val]
            avg = sum(scores) / len(scores)
            best = max(scores)
            print(f"    {val:>8}:  avg={avg:.4f}  best={best:.4f}  n={len(scores)}")

    # ── Baseline comparison ──
    print(f"\n\n{'='*80}")
    print("Baseline Comparison (kl=0.1, rw=1.0, dp=0.2, lr=0.001):")
    print(f"{'='*80}")
    baseline_name = "kl0.1_rw1.0_dp0.2_lr0.001"
    baseline = None
    for r in records:
        if r['name'] == baseline_name:
            baseline = r
            break
    if baseline:
        print(f"  Full={baseline['full']:.4f}  MissT={baseline['miss_t']:.4f}")
        # Count how many beat baseline
        better = [r for r in records if r['miss_t'] is not None and baseline['miss_t'] is not None and r['miss_t'] > baseline['miss_t']]
        worse = [r for r in records if r['miss_t'] is not None and baseline['miss_t'] is not None and r['miss_t'] < baseline['miss_t']]
        print(f"  Better than baseline: {len(better)}/{len(records)}")
        print(f"  Worse than baseline:  {len(worse)}/{len(records)}")
        if better:
            print(f"  Best improvement: +{max(r['miss_t'] - baseline['miss_t'] for r in better):.4f}")
    else:
        print("  Baseline not found in results.")

if __name__ == '__main__':
    main()
