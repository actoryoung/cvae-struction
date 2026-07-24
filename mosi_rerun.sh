#!/bin/bash
# MOSI re-run: 4 configs × 3 seeds = 12 runs
# Verify previous Table 2 data
set -e

PROJ=/home/ly/stu_work/projects/missing-modality-msa
cd "$PROJ"
DATA="casp_dataset/mosi.pkl"
EP=30; BS=32
LOG_DIR="/tmp/mosi_rerun"
CKPT_DIR="checkpoints/mosi_rerun"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "=== MOSI Re-run: 4 configs × 3 seeds = 12 runs ==="
echo "Started: $(date)"
echo ""

run_one() {
    local MODE=$1 KL=$2 SEED=$3 DESC=$4
    local NAME="mosi_${MODE}_kl${KL}_seed${SEED}"
    local LOG="${LOG_DIR}/${NAME}.txt"
    local CKPT="${CKPT_DIR}/${NAME}.pt"

    local FLAGS="--mode $MODE --dataset mosi --datapath $DATA"
    FLAGS="$FLAGS --num_epochs $EP --batch_size $BS --num_workers 0"
    FLAGS="$FLAGS --dropout_prob 0.2 --lr 0.001 --recon_weight 1.0"
    FLAGS="$FLAGS --seed $SEED --log_interval 50"
    FLAGS="$FLAGS --name $CKPT"

    if [ "$MODE" = "cvae" ]; then
        FLAGS="$FLAGS --kl_weight $KL"
    fi

    echo "[$(date +%H:%M:%S)] $NAME  # $DESC"
    python3 -u train_cvae.py $FLAGS > "$LOG" 2>&1
    python3 record_results.py "$LOG" "$NAME" \
        --config=dataset=mosi --config=mode=$MODE --config=kl=$KL \
        --config=seed=$SEED --config=lr=0.001
    echo "  -> DONE"
}

# Run sequentially to avoid dataloader contention on single pickle file
for SEED in 666 20260113 20040169; do
    echo "--- Seed $SEED ---"
    run_one concat 0   $SEED "concat baseline"
    run_one cvae   0.001 $SEED "CVAE KL=0.001"
    run_one cvae   0.4   $SEED "CVAE KL=0.4"
    run_one cvae   0.8   $SEED "CVAE KL=0.8"
done

echo ""
echo "=== ALL DONE: $(date) ==="
echo "Results: $LOG_DIR/"
echo "Checkpoints: $CKPT_DIR/"