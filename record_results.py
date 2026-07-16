#!/usr/bin/env python3
"""
Post-training hook: parse training output and save results.

Produces TWO outputs:
  1. results/{task_name}.json  — complete data for plotting (per-epoch curves + final test)
  2. stdout markdown snippet   — core metrics for EXPERIMENTS.md

Usage: python record_results.py <output_file> <task_name> [--config key=val ...]
"""

import sys, os, re, json
from datetime import datetime

EXP_DIR = "/home/ly/stu_work/projects/missing-modality-msa"
RESULTS_DIR = os.path.join(EXP_DIR, "results")


def parse_per_epoch(text):
    """Extract per-epoch validation metrics."""
    epochs = []
    pattern = r'Epoch\s+(\d+)\s+\|.*?Val L1\s+([\d.na]+)'
    for m in re.finditer(pattern, text):
        epochs.append({
            'epoch': int(m.group(1)),
            'val_l1': float(m.group(2)) if m.group(2) != 'nan' else None,
        })

    # Also extract acc-2 per epoch (printed separately as "Best" lines)
    best_pattern = r'>>> Best \(Acc-2:\s+([\d.]+)\)'
    best_matches = list(re.finditer(best_pattern, text))
    epoch_matches = list(re.finditer(r'Epoch\s+(\d+)\s+\|', text))

    # Match best acc-2 updates to epochs
    acc2_by_epoch = {}
    best_idx = 0
    for em in epoch_matches:
        ep = int(em.group(1))
        # Check if there was a "Best" update before this epoch line
        # Simple approach: track running best
        pass

    # Simpler: just extract all epoch summaries with their metrics
    epoch_block = re.findall(
        r'Epoch\s+(\d+)\s+\| Time\s+([\d.]+)s \| Train\s+([\d.na]+) \| Val L1\s+([\d.na]+)',
        text
    )
    for ep, t, train, vl1 in epoch_block:
        ep = int(ep)
        epochs.append({
            'epoch': ep,
            'time_s': float(t),
            'train_loss': float(train) if train != 'nan' else None,
            'val_l1': float(vl1) if vl1 != 'nan' else None,
        })

    # Remove duplicates (from the two-pass parsing)
    seen = set()
    unique = []
    for e in epochs:
        if e['epoch'] not in seen:
            seen.add(e['epoch'])
            unique.append(e)
    return sorted(unique, key=lambda x: x['epoch'])


def parse_final_test(text):
    """Extract final test metrics (4 settings × 4 metrics)."""
    results = {}
    section_match = re.search(r'FINAL TEST.*?\n(.*)', text, re.DOTALL)
    if not section_match:
        return results

    section = section_match.group(0)
    for miss in ['Full', 'Missing text', 'Missing audio', 'Missing vision']:
        pattern = rf'\[{miss}\]\n(.*?)(?:\n\n|\Z|\n\[)'
        m = re.search(pattern, section, re.DOTALL)
        if not m:
            continue
        block = m.group(1)
        results[miss] = {}
        for metric, regex in [
            ('Acc-2', r'Accuracy:\s+([\d.]+)'),
            ('F1', r'F1 score:\s+([\d.]+)'),
            ('MAE', r'MAE:\s+([\d.]+)'),
            ('Acc-7', r'mult_acc_7:\s+([\d.]+)'),
        ]:
            rm = re.search(regex, block)
            if rm:
                results[miss][metric] = float(rm.group(1))
    return results


def parse_best(text):
    """Extract best epoch and val acc-2."""
    best = re.search(r'best ep (\d+), Acc-2:\s+([\d.]+)', text)
    if best:
        return int(best.group(1)), float(best.group(2))
    return None, None


def main():
    if len(sys.argv) < 2:
        print("Usage: record_results.py <output_file> <task_name> [--config k=v ...]")
        sys.exit(1)

    output_file = sys.argv[1]
    task_name = sys.argv[2] if len(sys.argv) > 2 else "experiment"

    # Parse optional config
    config = {}
    for arg in sys.argv[3:]:
        if arg.startswith('--config') and '=' in arg:
            kv = arg.split('=', 1)[1] if '=' in arg else ''
            if '=' in kv:
                k, v = kv.split('=', 1)
                config[k] = v

    with open(output_file) as f:
        text = f.read()

    # Parse
    per_epoch = parse_per_epoch(text)
    final_test = parse_final_test(text)
    best_epoch, best_acc2 = parse_best(text)

    if best_acc2 is None:
        print("ERROR: Could not parse best Acc-2 from output")
        sys.exit(1)

    # ── Output 1: Complete JSON for plotting ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    record = {
        'task': task_name,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'config': config,
        'best_epoch': best_epoch,
        'best_val_acc2': best_acc2,
        'per_epoch': per_epoch,
        'final_test': final_test,
    }
    json_path = os.path.join(RESULTS_DIR, f'{task_name}.json')
    with open(json_path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"Full data saved to {json_path}")

    # ── Output 2: Core metrics table for EXPERIMENTS.md ──
    print(f"\n{'='*60}")
    print(f"Core metrics for EXPERIMENTS.md:")
    print(f"  Best epoch: {best_epoch}, Best val Acc-2: {best_acc2:.4f}")
    for miss in ['Full', 'Missing text', 'Missing audio', 'Missing vision']:
        if miss in final_test:
            r = final_test[miss]
            acc2 = r.get('Acc-2', '?')
            f1 = r.get('F1', '?')
            mae = r.get('MAE', '?')
            print(f"  {miss:15s} | Acc-2={acc2:.4f} | F1={f1:.4f} | MAE={mae:.4f}")

    # Generate markdown table row for EXPERIMENTS.md
    full_acc = final_test.get('Full', {}).get('Acc-2', 0)
    miss_t = final_test.get('Missing text', {}).get('Acc-2', 0)
    miss_a = final_test.get('Missing audio', {}).get('Acc-2', 0)
    miss_v = final_test.get('Missing vision', {}).get('Acc-2', 0)

    print(f"\n--- Copy this line to EXPERIMENTS.md ---")
    print(f"| {task_name} | {full_acc:.4f} | {miss_t:.4f} | {miss_a:.4f} | {miss_v:.4f} |")


if __name__ == '__main__':
    main()
