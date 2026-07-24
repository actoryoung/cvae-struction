#!/bin/bash
# Quick training status — run anytime: bash status.sh
cd /home/ly/stu_work/projects/missing-modality-msa

NOW=$(date +%H:%M:%S)
ALL_PROCS=$(ps aux | grep -E "train_(cvae|missmodal|gmd|t2dr|tmdc)" | grep -v grep | wc -l)

echo "=== Training @ $NOW (${ALL_PROCS} processes) ==="
echo ""

python3 << 'PYEOF'
import glob, re, os, subprocess, numpy as np

def parse_log(fpath):
    try:
        with open(fpath) as f:
            content = f.read()
    except:
        return False, None, None, None, None

    has_final = "FINAL TEST" in content

    # Extract final test metrics
    fl_match = re.search(r'\[Full\].*?Accuracy:\s+([\d.]+)', content, re.DOTALL)
    mt_match = re.search(r'Missing text.*?Accuracy:\s+([\d.]+)', content, re.DOTALL)
    ma_match = re.search(r'Missing audio.*?Accuracy:\s+([\d.]+)', content, re.DOTALL)
    mv_match = re.search(r'Missing vision.*?Accuracy:\s+([\d.]+)', content, re.DOTALL)
    fl = float(fl_match.group(1)) if fl_match else None
    mt = float(mt_match.group(1)) if mt_match else None
    ma = float(ma_match.group(1)) if ma_match else None
    mv = float(mv_match.group(1)) if mv_match else None

    # Current epoch & best val acc
    ep_match = re.findall(r'Epoch\s+(\d+)\s+\|', content)
    best_match = re.findall(r'>>> Best \(Acc-2:\s+([\d.]+)\)', content)
    ep = ep_match[-1] if ep_match else None
    best_acc = best_match[-1] if best_match else None

    return has_final, fl, mt, ma, mv, ep, best_acc


def show_sweep(directory, label, sort_by_mt=True):
    pattern = os.path.join(directory, "*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        return

    done = []
    running = []
    for f in files:
        name = os.path.basename(f).replace(".txt", "")
        has_final, fl, mt, ma, mv, ep, best = parse_log(f)
        if has_final:
            done.append((name, fl or 0, mt or 0, ma or 0, mv or 0))
        else:
            running.append((name, ep, best))

    total = len(files)
    print(f"── {label} ({len(done)}/{total} done) ──")

    # Completed
    if done:
        for name, fl, mt, ma, mv in sorted(done, key=lambda x: -x[2] if sort_by_mt else -x[0]):
            print(f"  ✅ {name:<55s} Full={fl:.4f}  MissT={mt:.4f}  MissA={ma:.4f}  MissV={mv:.4f}")

    # Running
    if running:
        if done:
            print("")
        for name, ep, best in sorted(running, key=lambda x: x[0]):
            ep_str = f"{ep:>2s}" if ep else " ?"
            best_str = f" best={best}" if best else ""
            print(f"  🔄 {name:<55s} epoch {ep_str}/30{best_str}")

    print("")

    return done  # Return for aggregation


def aggregate_method(directory, method_name):
    """Aggregate results across seeds, return (name, full_mean, mt_mean, mt_std, ...)"""
    pattern = os.path.join(directory, "*.txt")
    files = sorted(glob.glob(pattern))
    if len(files) < 1:
        return None
    results = []
    for f in files:
        has_final, fl, mt, ma, mv, _, _ = parse_log(f)
        if has_final and mt is not None:
            results.append((fl or 0, mt or 0, ma or 0, mv or 0))
    if not results:
        return None
    fls, mts, mas, mvs = zip(*results)
    return (method_name,
            np.mean(fls), np.std(fls),
            np.mean(mts), np.std(mts),
            np.mean(mas), np.std(mas),
            np.mean(mvs), np.std(mvs),
            len(results))


# ═══════════════════════════════════════════════════════════
# Internal experiments
# ═══════════════════════════════════════════════════════════

print("═══ Internal Experiments ═══")
show_sweep("/tmp/det_mlp",   "DetMLP (fusion-space deterministic)")
show_sweep("/tmp/raw_mlp",   "RawMLP (raw-feature-space 768d)")
show_sweep("/tmp/subsample", "MOSEI Subsampling (12 runs)")
show_sweep("/tmp/ablation_recon", "Ablation: Recon=0 CVAE")

# ═══════════════════════════════════════════════════════════
# MOSI re-run (dataset-scale validation)
# ═══════════════════════════════════════════════════════════

print("═══ MOSI Re-run (4 configs × 3 seeds) ═══")
show_sweep("/tmp/mosi_rerun", "MOSI (1.3K samples)")

# Aggregate MOSI by config
def aggregate_mosi():
    """Group MOSI results by config name."""
    import collections
    pattern = os.path.join("/tmp/mosi_rerun", "*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        return
    groups = collections.defaultdict(list)
    for f in files:
        has_final, fl, mt, ma, mv, _, _ = parse_log(f)
        if has_final and mt is not None:
            name = os.path.basename(f).replace(".txt", "")
            # Extract config: "mosi_concat_kl0_seed666" → "concat"
            parts = name.split("_seed")[0].replace("mosi_", "")
            groups[parts].append((fl or 0, mt or 0, ma or 0, mv or 0))
    if groups:
        print(f"\n  {'Config':<22s} {'Full':>10s} {'MissT':>12s} {'MissA':>12s} {'MissV':>12s}  N")
        print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*12} {'-'*12}  ---")
        for cfg in sorted(groups.keys()):
            vals = groups[cfg]
            fls, mts, mas, mvs = zip(*vals)
            print(f"  {cfg:<22s} {np.mean(fls):.4f}±{np.std(fls):<6.4f} {np.mean(mts):.4f}±{np.std(mts):<6.4f} {np.mean(mas):.4f}±{np.std(mas):<6.4f} {np.mean(mvs):.4f}±{np.std(mvs):<6.4f}  {len(vals)}")

aggregate_mosi()

# ═══════════════════════════════════════════════════════════
# External baselines
# ═══════════════════════════════════════════════════════════

print("\n═══ External Baselines ═══")
b_missmodal = show_sweep("/tmp/missmodal", "MissModal (TACL 2023)")
b_gmd       = show_sweep("/tmp/gmd",       "GMD (AAAI 2024)")
b_tmdc       = show_sweep("/tmp/tmdc",       "TMDC (AAAI 2026)")

# ═══════════════════════════════════════════════════════════
# Aggregate comparison table
# ═══════════════════════════════════════════════════════════

print("═══ Aggregate Comparison — MOSEI (mean ± std) ═══")
methods = []
methods.append(aggregate_method("/tmp/missmodal", "MissModal  "))
methods.append(aggregate_method("/tmp/gmd",       "GMD        "))
methods.append(aggregate_method("/tmp/tmdc",       "TMDC       "))
methods.append(aggregate_method("/tmp/det_mlp",   "DetMLP     "))
methods.append(aggregate_method("/tmp/raw_mlp",   "RawMLP     "))

if any(m is not None for m in methods):
    print(f"  {'Method':<14s} {'Full':>10s} {'MissT':>12s} {'MissA':>12s} {'MissV':>12s}  N")
    print(f"  {'-'*14} {'-'*10} {'-'*12} {'-'*12} {'-'*12}  ---")
    for m in methods:
        if m is None:
            continue
        name, fm, fs, mm, ms, am, as_, vm, vs, n = m
        print(f"  {name:<14s} {fm:.4f}±{fs:<6.4f} {mm:.4f}±{ms:<6.4f} {am:.4f}±{as_:<6.4f} {vm:.4f}±{vs:<6.4f}  {n}")

# ═══════════════════════════════════════════════════════════
# Process check
# ═══════════════════════════════════════════════════════════

print("")
try:
    out = subprocess.check_output(["ps", "aux"], text=True)
    scripts = {
        'train_cvae': 'CVAE',
        'train_missmodal': 'MissModal',
        'train_gmd': 'GMD',
        'train_tmdc': 'TMDC',
    }
    active = []
    for line in out.split('\n'):
        if 'grep' in line:
            continue
        for script, label in scripts.items():
            if script in line:
                m = re.search(r'--seed\s+(\d+)', line)
                seed = m.group(1) if m else '?'
                mode_m = re.search(r'--modes+(S+)', line)
                ds_m = re.search(r'--datasets+(S+)', line)
                extra = ""
                if mode_m: extra += f"/{mode_m.group(1)}"
                if ds_m: extra += f"@{ds_m.group(1)}"
                active.append(f"{label}{extra}(s={seed})")
    if active:
        print(f"  {len(active)} processes: {', '.join(sorted(active))}")
    else:
        print("  No training running.")
except:
    print("  Could not check processes.")
PYEOF
