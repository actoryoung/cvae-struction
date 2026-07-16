#!/bin/bash
# ============================================================
# Smart 30-run exploratory sweep
# Design: Blocked fractional design focusing on key interactions
# 6 parallel × 5 batches ≈ 10 hours
# ============================================================
set +e

PROJ_DIR="/home/ly/stu_work/projects/missing-modality-msa"
cd "$PROJ_DIR"

DATA="casp_dataset/mosei_pt"
EPOCHS=30; BS=32; WORKERS=0; SEED=666; N_PARALLEL=6

echo "============================================================"
echo "Smart Sweep: 30 hand-picked combinations, 6 parallel"
echo "============================================================"
echo "Started at: $(date)"
echo ""

# ─── Build curated list ───
declare -a CMDS
COUNT=0

run_one() {
    local KL=$1 RW=$2 DP=$3 LR=$4 DESC=$5
    COUNT=$((COUNT + 1))
    local NAME="kl${KL}_rw${RW}_dp${DP}_lr${LR}"
    local LOG="/tmp/sweep_grid/${NAME}.txt"
    local CKPT="checkpoints/sweep/${NAME}.pt"
    CMD="echo '[$COUNT/30] $(date +%H:%M:%S) ${NAME}  # ${DESC}' && \
         PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py \
           --mode cvae --dataset mosei --datapath $DATA --use_pt \
           --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS \
           --kl_weight $KL --recon_weight $RW --dropout_prob $DP --lr $LR \
           --seed $SEED --name $CKPT --log_interval 500 \
           2>&1 | tee $LOG && \
         /usr/bin/python3 record_results.py $LOG $NAME \
           --config=kl_weight=$KL --config=recon_weight=$RW \
           --config=dropout_prob=$DP --config=lr=$LR \
           --config=mode=cvae --config=batch_size=$BS --config=epochs=$EPOCHS"
    CMDS+=("$CMD")
}

# ── Block 1: KL×Recon core interaction grid (12 runs) ──
# The most critical interaction — they directly trade off in the loss
# KL: 0.05, 0.1, 0.2, 0.5   Recon: 0.5, 1.0, 2.0
# Fixed: dp=0.2, lr=0.001
echo "Block 1: KL×Recon grid (12 runs)"
for KL in 0.05 0.1 0.2 0.5; do
  for RW in 0.5 1.0 2.0; do
    run_one $KL $RW 0.2 0.001 "KL-Recon core"
  done
done

# ── Block 2: Dropout sweep around baseline (6 runs) ──
# Baseline (kl=0.1, rw=1.0) + 2 nearby recon values × 3 dropouts
# Tests: does dropout interact with reconstruction weight?
echo "Block 2: Dropout sweep (6 runs)"
for RW in 0.5 1.0 2.0; do
  for DP in 0.1 0.3; do  # 0.2 covered in block 1
    run_one 0.1 $RW $DP 0.001 "dropout sweep"
  done
done

# ── Block 3: LR sweep around baseline (6 runs) ──
echo "Block 3: LR sweep (6 runs)"
for RW in 0.5 1.0 2.0; do
  for LR in 0.0008 0.002; do  # 0.001 covered in block 1
    run_one 0.1 $RW 0.2 $LR "LR sweep"
  done
done

# ── Block 4: Edge cases (6 runs) ──
echo "Block 4: Edge cases (6 runs)"
# Very high recon weight — does it force better reconstruction?
run_one 0.05 5.0 0.2 0.001 "low-KL high-recon"
run_one 0.1  5.0 0.2 0.001 "mid-KL high-recon"
run_one 0.2  5.0 0.2 0.001 "high-KL high-recon"
# Interesting corner combinations
run_one 0.2 0.5 0.3 0.002   "high-KL low-recon high-DP high-LR"
run_one 0.5 2.0 0.1 0.0008  "vhigh-KL high-recon low-DP low-LR"
run_one 0.05 1.0 0.3 0.002  "low-KL high-DP high-LR"

echo ""
echo "Total: ${#CMDS[@]} combinations"

# ─── Launch ───
rm -f /tmp/sweep_grid/*.txt results/kl*.json checkpoints/sweep/*.pt

RUNNING=0
for i in "${!CMDS[@]}"; do
  if [ $RUNNING -ge $N_PARALLEL ]; then
    wait -n
    RUNNING=$((RUNNING - 1))
  fi
  eval "${CMDS[$i]}" &
  RUNNING=$((RUNNING + 1))
done
wait

echo ""
echo "============================================================"
echo "ALL 30 JOBS COMPLETED at $(date)"
echo "Run: python3 summarize_sweep.py"
echo "============================================================"
