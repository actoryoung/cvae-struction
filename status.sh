#!/bin/bash
# Quick training status — run anytime: bash status.sh
cd /home/ly/stu_work/projects/missing-modality-msa

NOW=$(date +%H:%M:%S)
RUNNING=$(ps aux | grep train_cvae | grep -v grep | wc -l)

echo "=== Training @ $NOW (${RUNNING} processes) ==="
echo ""

python3 << 'PYEOF'
import glob, re, os

def parse_log(fpath):
    with open(fpath) as f:
        content = f.read()
    has_final = "FINAL TEST" in content

    # Extract metrics
    fl_match = re.search(r'\[Full\].*?Accuracy:\s+([\d.]+)', content, re.DOTALL)
    mt_match = re.search(r'Missing text.*?Accuracy:\s+([\d.]+)', content, re.DOTALL)
    fl = float(fl_match.group(1)) if fl_match else None
    mt = float(mt_match.group(1)) if mt_match else None

    # Current epoch
    eps = re.findall(r'Epoch\s+(\d+)\s+\|', content)
    vls = re.findall(r'Val L1\s+([\d.]+)', content)
    ep = eps[-1] if eps else None
    vl = vls[-1] if vls else None

    return has_final, fl, mt, ep, vl

def show_sweep(directory, label):
    pattern = os.path.join(directory, "*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        return

    done = []
    running = []
    for f in files:
        name = os.path.basename(f).replace(".txt", "")
        has_final, fl, mt, ep, vl = parse_log(f)
        if has_final:
            done.append((mt or 0, name, fl or 0))
        else:
            running.append((name, ep, vl))

    total = len(files)
    print(f"── {label} ({len(done)}/{total} done) ──")

    # Completed: sort by MissT descending
    for mt, name, fl in sorted(done, key=lambda x: -x[0]):
        print(f"  ✅ {name:<50s} Full={fl:.4f}  MissT={mt:.4f}")

    # Running
    if running:
        if done:
            print("")
        for name, ep, vl in sorted(running, key=lambda x: x[0]):
            ep_str = f"{ep:>2s}" if ep else " ?"
            vl_str = f"{vl}" if vl else "?"
            print(f"  🔄 {name:<50s} epoch {ep_str}/30  ValL1={vl_str}")

    print("")

show_sweep("/tmp/multi_seed", "Multi-Seed (seeds 20260113, 20040169)")
show_sweep("/tmp/mosi_test", "MOSI Test")
show_sweep("/tmp/mosi_kl", "MOSI KL")
import subprocess
try:
    out = subprocess.check_output(["ps", "aux"], text=True)
    running = len([l for l in out.split('\n') if 'train_cvae' in l and 'grep' not in l])
except:
    running = 0

if running == 0:
    print("No training running. Start: bash combo_sweep.sh")
PYEOF
