#!/usr/bin/env python3
"""Generate paper figures: Fig 1 (β-U curve) + Fig 2 (benchmark comparison)."""
import json, os, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS_DIR = "/home/ly/stu_work/projects/missing-modality-msa/results"
OUT_DIR = "/home/ly/stu_work/projects/missing-modality-msa/paper/figures"

def load_sweep():
    records = []
    for fname in sorted(os.listdir(os.path.join(RESULTS_DIR, "sweep"))):
        if not fname.endswith('.json'): continue
        with open(os.path.join(RESULTS_DIR, "sweep", fname)) as f:
            r = json.load(f)
        cfg = r['config']
        ft = r.get('final_test', {})
        records.append({
            'kl': float(cfg.get('kl_weight', 0)),
            'rw': float(cfg.get('recon_weight', 0)),
            'miss_t': ft.get('Missing text', {}).get('Acc-2'),
        })
    return records

def load_external():
    baselines = {}
    labels = {'missmodal': 'MissModal\n(TACL 2023)', 'gmd': 'GMD\n(AAAI 2024)', 'tmdc': 'TMDC\n(AAAI 2026)'}
    colors = {'missmodal': '#e74c3c', 'gmd': '#f39c12', 'tmdc': '#9b59b6'}
    for method in ['missmodal', 'gmd', 'tmdc']:
        mts, fls = [], []
        for fname in glob.glob(f"/tmp/{method}/*.txt"):
            with open(fname) as f: text = f.read()
            mt = re.search(r'\[Missing text\].*?Accuracy:\s+([\d.]+)', text, re.DOTALL)
            if mt: mts.append(float(mt.group(1)))
        if mts:
            baselines[method] = {
                'name': labels[method], 'color': colors[method],
                'mt_mean': np.mean(mts), 'mt_std': np.std(mts), 'n': len(mts),
            }
    return baselines

# ── Paper values ──
CONCAT = 0.560
DETMLP = 0.588

def fig1_beta_u_curve(sweep):
    """Figure 1: β-U curve with phase transition annotation."""
    fig, ax = plt.subplots(figsize=(7, 4.8))

    rw_vals = sorted(set(r['rw'] for r in sweep if r['miss_t'] is not None))
    cmap = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(rw_vals)))
    for rw, c in zip(rw_vals, cmap):
        pts = [(r['kl'], r['miss_t']) for r in sweep if r['miss_t'] and r['rw'] == rw]
        if pts:
            kls, mts = zip(*sorted(pts))
            ax.scatter(kls, mts, c=[c], edgecolors='white', s=55, zorder=3, alpha=0.7,
                      label=f'λ={rw}')

    # Multi-seed error bars
    ax.errorbar([0.4, 0.8], [0.578, 0.586], yerr=[0.012, 0.006],
               fmt='o-', color='#1a1a2e', linewidth=2.5, markersize=11,
               capsize=6, capthick=2, zorder=5, label='CVAE (3 seeds)')

    # DetMLP scatter point
    ax.scatter([0], [DETMLP], marker='D', s=100, color='#2ecc71', edgecolors='#1a1a2e',
              linewidths=1.5, zorder=6, label=f'DetMLP (no KL, {DETMLP:.3f})')

    # Reference: concat
    ax.axhline(y=CONCAT, color='#e74c3c', linestyle='--', lw=1.3, alpha=0.7,
              zorder=2, label=f'Concat ({CONCAT:.3f})')

    # Shaded regions
    ax.axvspan(0, 0.25, alpha=0.07, color='#e74c3c', zorder=0)
    ax.axvspan(0.35, 1.05, alpha=0.07, color='#27ae60', zorder=0)
    ax.text(0.1, 0.615, 'Standard\nVAE regime', ha='center', fontsize=8.5,
           color='#c0392b', style='italic')
    ax.text(0.65, 0.615, 'Optimal regime (β ≥ 0.4)', ha='center', fontsize=8.5,
           color='#27ae60', style='italic', fontweight='bold')

    ax.annotate('+5.8pp\nphase transition', xy=(0.33, 0.577), xytext=(0.52, 0.55),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
               fontsize=8.5, color='#2c3e50', fontweight='bold', ha='center', va='top')

    ax.set_xlabel('KL weight β', fontsize=12)
    ax.set_ylabel('Missing-text Acc-2', fontsize=12)
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9, ncol=2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.48, 0.63)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'fig1_beta_u_curve.{fmt}'),
                   dpi=200, bbox_inches='tight', facecolor='white')
    print("Figure 1 saved: β-U curve")


def fig5a_kl_recon_heatmap(sweep):
    """Figure 5a: KL × Reconstruction Weight interaction heatmap."""
    from matplotlib.colors import LinearSegmentedColormap

    # Build pivot grid
    kls = sorted(set(r['kl'] for r in sweep if r['miss_t'] is not None))
    rws = sorted(set(r['rw'] for r in sweep if r['miss_t'] is not None))
    # Filter to meaningful RW range (exclude 5.0 outlier)
    rws = [rw for rw in rws if rw <= 2.0]

    grid = np.full((len(rws), len(kls)), np.nan)
    for i, rw in enumerate(rws):
        for j, kl in enumerate(kls):
            vals = [r['miss_t'] for r in sweep
                    if r['miss_t'] is not None and r['kl'] == kl and r['rw'] == rw]
            if vals:
                grid[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(6, 4.2))

    # Custom colormap: red (low) → yellow → green (high)
    colors_list = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
    cmap = LinearSegmentedColormap.from_list('missT', colors_list, N=256)

    im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0.49, vmax=0.59,
                   origin='upper')

    # Annotate each cell
    for i in range(len(rws)):
        for j in range(len(kls)):
            if not np.isnan(grid[i, j]):
                text = ax.text(j, i, f'{grid[i, j]:.3f}',
                              ha='center', va='center', fontsize=12,
                              fontweight='bold',
                              color='white' if grid[i, j] < 0.545 else '#1a1a2e')

    ax.set_xticks(range(len(kls)))
    ax.set_xticklabels([f'β={k}' for k in kls], fontsize=11)
    ax.set_yticks(range(len(rws)))
    ax.set_yticklabels([f'λ={rw}' for rw in rws], fontsize=11)
    ax.set_xlabel('KL weight β', fontsize=13)
    ax.set_ylabel('Reconstruction weight λ', fontsize=13)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Missing-text Acc-2', fontsize=11)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'fig5a_kl_recon_heatmap.{fmt}'),
                   dpi=200, bbox_inches='tight', facecolor='white')
    print("Figure 5a saved: KL×Recon heatmap")


def fig5b_kl_recon_lines(sweep):
    """Figure 5b: MissT vs Reconstruction Weight, per KL value."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    kl_vals = sorted(set(r['kl'] for r in sweep if r['miss_t'] is not None))
    rw_vals = sorted(set(r['rw'] for r in sweep if r['miss_t'] is not None))
    rw_vals = [rw for rw in rw_vals if rw <= 2.0]  # exclude 5.0 outlier

    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    markers = ['o', 's', '^', 'D']

    for kl, color, marker in zip(kl_vals, colors, markers):
        xs, ys, yerrs = [], [], []
        for rw in rw_vals:
            vals = [r['miss_t'] for r in sweep
                    if r['miss_t'] is not None and r['kl'] == kl and r['rw'] == rw]
            if vals:
                xs.append(rw)
                ys.append(np.mean(vals))
                yerrs.append(np.std(vals) if len(vals) > 1 else 0)

        if xs:
            ax.errorbar(xs, ys, yerr=yerrs, fmt=f'{marker}-',
                       color=color, linewidth=2.2, markersize=10,
                       capsize=5, capthick=1.8,
                       label=f'β={kl}', alpha=0.9)

    # Reference: concat baseline
    ax.axhline(y=CONCAT, color='#95a5a6', linestyle='--', lw=1.3, alpha=0.7,
              label=f'Concat ({CONCAT:.3f})')

    ax.set_xlabel('Reconstruction weight λ', fontsize=13)
    ax.set_ylabel('Missing-text Acc-2', fontsize=13)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.3, 2.2)
    ax.set_ylim(0.48, 0.60)

    # Annotation: optimal region
    ax.annotate('Optimal: β=0.5, λ=1.0\n(MissT=0.585)',
               xy=(1.0, 0.5853), xytext=(1.4, 0.57),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
               fontsize=9, color='#2c3e50', fontweight='bold')

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'fig5b_kl_recon_lines.{fmt}'),
                   dpi=200, bbox_inches='tight', facecolor='white')
    print("Figure 5b saved: KL×Recon line plot")


def fig7_benchmark(ext):
    """Figure 2: Benchmark comparison bar chart."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    methods = [
        ('MissModal (TACL 2023)',   0.516, 0.008, '#e74c3c'),
        ('Concat (zero-fill)',      CONCAT, 0.011, '#bdc3c7'),
        ('RawMLP (768d)',            0.576, 0.014, '#95a5a6'),
        ('GMD (AAAI 2024)',          0.581, 0.019, '#f39c12'),
        ('CVAE β=0.8',              0.586, 0.006, '#2980b9'),
        ('TMDC (AAAI 2026)',         0.586, 0.007, '#9b59b6'),
        ('DetMLP (ours)',            DETMLP, 0.015, '#2ecc71'),
    ]
    methods.sort(key=lambda x: x[1])

    names = [m[0] for m in methods]
    mts   = [m[1] for m in methods]
    errs  = [m[2] for m in methods]
    colors = [m[3] for m in methods]

    bars = ax.barh(range(len(methods)), mts, xerr=errs, color=colors,
                   edgecolor='white', height=0.6, capsize=4, alpha=0.92)

    for i, (mt, err) in enumerate(zip(mts, errs)):
        ax.text(mt + err + 0.004, i, f'{mt:.3f}', va='center', fontsize=9.5, fontweight='bold')

    best_idx = max(range(len(methods)), key=lambda i: methods[i][1])
    bars[best_idx].set_edgecolor('#1a1a2e')
    bars[best_idx].set_linewidth(2.5)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel('Missing-text Acc-2', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_xlim(0.50, 0.64)

    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.9, label='Ours (fusion-space reconstruction)'),
        Patch(facecolor='#bdc3c7', alpha=0.9, label='Passive baseline'),
        Patch(facecolor='#9b59b6', alpha=0.9, label='External (re-implemented)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'fig7_benchmark.{fmt}'),
                   dpi=200, bbox_inches='tight', facecolor='white')
    print("Figure 2 saved: benchmark comparison")


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    sweep = load_sweep()
    ext = load_external()
    fig1_beta_u_curve(sweep)
    fig5a_kl_recon_heatmap(sweep)
    fig5b_kl_recon_lines(sweep)
    fig7_benchmark(ext)
    print("Done.")
