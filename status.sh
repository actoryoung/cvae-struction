#!/bin/bash
# Quick training status — run anytime: bash status.sh
cd /home/ly/stu_work/projects/missing-modality-msa

NOW=$(date +%H:%M:%S)
RUNNING=$(ps aux | grep train_cvae | grep -v grep | wc -l)

echo "=== Training @ $NOW (${RUNNING} processes) ==="
echo ""

show_sweep() {
  local DIR=$1 LABEL=$2
  local DONE=$(grep -l 'FINAL TEST' ${DIR}/*.txt 2>/dev/null | wc -l)
  local TOTAL=$(ls ${DIR}/*.txt 2>/dev/null | wc -l)
  local RUNNING=$(grep -L 'FINAL TEST' ${DIR}/*.txt 2>/dev/null | wc -l)

  if [ $TOTAL -eq 0 ]; then return; fi

  echo "── ${LABEL} (${DONE}/${TOTAL} done) ──"

  if [ $RUNNING -gt 0 ]; then
    for f in $(ls -t ${DIR}/*.txt 2>/dev/null); do
      if grep -q 'FINAL TEST' "$f" 2>/dev/null; then continue; fi
      name=$(basename "$f" .txt)
      ep=$(grep -oP 'Epoch\s+\d+' "$f" 2>/dev/null | tail -1 | awk '{print $2}')
      vl=$(grep -oP 'Val L1\s+([\d.]+)' "$f" 2>/dev/null | tail -1 | awk '{print $3}')
      printf "  %-50s epoch %2s/30  ValL1=%s\n" "$name" "$ep" "$vl"
    done
  else
    # All done — show top 3 by MissT
    for f in $(grep -l 'FINAL TEST' ${DIR}/*.txt 2>/dev/null); do
      name=$(basename "$f" .txt)
      mt=$(grep -A 10 "Missing text" "$f" | grep "Accuracy:" | head -1 | awk '{print $2}')
      fl=$(grep -A 10 "\[Full\]" "$f" | grep "Accuracy:" | head -1 | awk '{print $2}')
      echo "  ${name}=${mt}" >> /tmp/.sweep_sort_$$
    done
    echo "  (all done, top MissT:)"
    sort -t'=' -k2 -rn /tmp/.sweep_sort_$$ 2>/dev/null | head -3 | while IFS='=' read name mt; do
      printf "    %-48s MissT=%s\n" "$name" "$mt"
    done
    rm -f /tmp/.sweep_sort_$$
  fi
  echo ""
}

show_sweep "/tmp/sweep_grid" "Smart Sweep"
show_sweep "/tmp/combo_sweep" "Combo Sweep"

if [ $RUNNING -eq 0 ]; then
  echo "No training running. Start: bash combo_sweep.sh"
fi
