#!/bin/bash
# ============================================================
# Experiment B: Raw-Feature-Space MLP Reconstruction
# Tests: Is fusion-space (40d) reconstruction better than
#        raw-feature-space (text 768d) reconstruction?
# 3 seeds × 1 config = 3 MOSEI runs (~3h each, 3 parallel)
# ============================================================
set +e

PROJ_DIR="/home/ly/stu_work/projects/missing-modality-msa"
cd "$PROJ_DIR"

DATA="casp_dataset/mosei_pt"
EPOCHS=30; BS=32; WORKERS=0; N_PARALLEL=3
LOG_DIR="/tmp/raw_mlp"
CKPT_DIR="checkpoints/raw_mlp"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================================"
echo "Experiment B: Raw-Feature-Space MLP Reconstruction"
echo "3 seeds × 1 config = 3 MOSEI runs"
echo "============================================================"
echo "Started at: $(date)"
echo ""

declare -a CMDS
COUNT=0

run_one() {
    local SEED=$1 DESC=$2
    COUNT=$((COUNT + 1))
    local NAME="raw_mlp_seed${SEED}"
    local LOG="${LOG_DIR}/${NAME}.txt"
    local CKPT="${CKPT_DIR}/${NAME}.pt"

    local FLAGS="--mode raw_mlp --dataset mosei --datapath $DATA --use_pt"
    FLAGS="$FLAGS --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS"
    FLAGS="$FLAGS --dropout_prob 0.2 --lr 0.001 --recon_weight 1.0"
    FLAGS="$FLAGS --seed $SEED --log_interval 500"
    FLAGS="$FLAGS --name $CKPT"

    local RCONFIG="--config=mode=raw_mlp --config=seed=$SEED"
    RCONFIG="$RCONFIG --config=lr=0.001 --config=batch_size=$BS --config=epochs=$EPOCHS"
    RCONFIG="$RCONFIG --config=dropout_prob=0.2 --config=recon_weight=1.0"

    local CMD="echo '[$COUNT/3] $(date +%H:%M:%S) ${NAME}  # ${DESC}' && \
         PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py $FLAGS \
           > $LOG 2>&1 && \
         /usr/bin/python3 record_results.py $LOG $NAME $RCONFIG"
    CMDS+=("$CMD")
}

echo "── RawMLP (3 seeds, 768d text bottleneck) ──"
for SEED in 666 20260113 20040169; do
    run_one $SEED "RawMLP seed=$SEED"
done

TOTAL=${#CMDS[@]}
echo ""
echo "Total: $TOTAL runs (~9 GPU-hours)"
echo ""

# Launch with N-way parallelism
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
echo "RAW-FEATURE MLP ALL DONE at $(date)"
echo "Results: $LOG_DIR/"
echo "============================================================"
