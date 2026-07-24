#!/bin/bash
# ============================================================
# Baseline Experiments: T2DR (ACL 2025 Findings)
# Tests: Does two-tier deficiency-resistant framework help missing-text?
# 3 seeds × 1 config = 3 MOSEI runs (~3.5h each, 3 parallel)
# ============================================================
set +e

PROJ_DIR="/home/ly/stu_work/projects/missing-modality-msa"
cd "$PROJ_DIR"

DATA="casp_dataset/mosei_pt"
EPOCHS=30; BS=32; WORKERS=0; N_PARALLEL=3
LOG_DIR="/tmp/t2dr"
CKPT_DIR="checkpoints/t2dr"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================================"
echo "Baseline: T2DR (ACL 2025 Findings)"
echo "3 seeds × 1 config = 3 MOSEI runs"
echo "============================================================"
echo "Started at: $(date)"
echo ""

declare -a CMDS
COUNT=0

run_one() {
    local SEED=$1 DESC=$2
    COUNT=$((COUNT + 1))
    local NAME="t2dr_seed${SEED}"
    local LOG="${LOG_DIR}/${NAME}.txt"
    local CKPT="${CKPT_DIR}/${NAME}.pt"

    local CMD="echo '[$COUNT/3] $(date +%H:%M:%S) ${NAME}  # ${DESC}' && \
         /usr/bin/python3 -u train_t2dr.py \
           --datapath $DATA --dataset mosei \
           --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS \
           --dropout_prob 0.2 --lr 0.001 --recon_weight 1.0 \
           --seed $SEED --log_interval 200 \
           --name $CKPT \
           > $LOG 2>&1 && \
         /usr/bin/python3 record_results.py $LOG $NAME \
           --config=method=t2dr --config=seed=$SEED \
           --config=lr=0.001 --config=recon=1.0"
    CMDS+=("$CMD")
}

run_one 666 "seed 666"
run_one 20260113 "seed 2"
run_one 20040169 "seed 3"

echo "Launching $N_PARALLEL parallel trainers..."
for i in $(seq 0 $((${#CMDS[@]} - 1))); do
    if (( i % N_PARALLEL == 0 )); then
        wait
    fi
    eval "${CMDS[$i]}" &
done
wait

echo ""
echo "============================================================"
echo "T2DR baseline done at $(date)"
echo "Results: $LOG_DIR/"
echo "============================================================"
