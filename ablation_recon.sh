#!/bin/bash
# ============================================================
# Ablation: Reconstruction Loss
# Tests: Does CVAE gain come from reconstruction or just KL?
# recon_weight=0 → no MSE reconstruction, only KL + regression
# 3 seeds × 1 config = 3 MOSEI runs (~2h each, 3 parallel ≈ 2h)
# ============================================================
set +e

PROJ_DIR="/home/ly/stu_work/projects/missing-modality-msa"
cd "$PROJ_DIR"

DATA="casp_dataset/mosei_pt"
EPOCHS=30; BS=32; WORKERS=0; N_PARALLEL=3
LOG_DIR="/tmp/ablation_recon"
CKPT_DIR="checkpoints/ablation_recon"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================================"
echo "Ablation: Recon=0 (CVAE KL=0.8, no reconstruction loss)"
echo "3 seeds × 1 config = 3 MOSEI runs"
echo "============================================================"
echo "Started at: $(date)"
echo ""

declare -a CMDS
COUNT=0

run_one() {
    local SEED=$1
    COUNT=$((COUNT + 1))
    local NAME="ab_recon0_seed${SEED}"
    local LOG="${LOG_DIR}/${NAME}.txt"
    local CKPT="${CKPT_DIR}/${NAME}.pt"

    local FLAGS="--mode cvae --dataset mosei --datapath $DATA --use_pt"
    FLAGS="$FLAGS --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS"
    FLAGS="$FLAGS --kl_weight 0.8 --recon_weight 0.0 --dropout_prob 0.2 --lr 0.001"
    FLAGS="$FLAGS --seed $SEED --log_interval 500"
    FLAGS="$FLAGS --name $CKPT"

    local RCONFIG="--config=mode=cvae --config=seed=$SEED"
    RCONFIG="$RCONFIG --config=kl_weight=0.8 --config=recon_weight=0.0"
    RCONFIG="$RCONFIG --config=lr=0.001 --config=batch_size=$BS --config=epochs=$EPOCHS"
    RCONFIG="$RCONFIG --config=dropout_prob=0.2"

    local CMD="echo '[$COUNT/3] $(date +%H:%M:%S) ${NAME}' && \
         PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py $FLAGS \
           > $LOG 2>&1 && \
         /usr/bin/python3 record_results.py $LOG $NAME $RCONFIG"
    CMDS+=("$CMD")
}

for SEED in 666 20260113 20040169; do
    run_one $SEED
done

TOTAL=${#CMDS[@]}
echo "Total: $TOTAL runs"
echo ""

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
echo "ABLATION RECON=0 ALL DONE at $(date)"
echo "Results: $LOG_DIR/"
echo "============================================================"
