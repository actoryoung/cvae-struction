#!/bin/bash
# ============================================================
# Experiment C: MOSEI Subsampling
# Tests: Does dataset size drive optimal KL preference?
# 4 sizes × 3 configs = 12 MOSEI runs (~6 GPU-hours total)
# ============================================================
set +e

PROJ_DIR="/home/ly/stu_work/projects/missing-modality-msa"
cd "$PROJ_DIR"

DATA="casp_dataset/mosei_pt"
EPOCHS=30; BS=32; WORKERS=0; SEED=666; N_PARALLEL=6
LOG_DIR="/tmp/subsample"
CKPT_DIR="checkpoints/subsample"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================================"
echo "Experiment C: MOSEI Subsampling"
echo "4 sizes × 3 configs = 12 runs"
echo "============================================================"
echo "Started at: $(date)"
echo ""

declare -a CMDS
COUNT=0

run_one() {
    local SIZE=$1 KL=$2 MODE=$3 DESC=$4
    COUNT=$((COUNT + 1))

    # Build name
    local NAME="sub${SIZE}"
    if [ "$MODE" = "concat" ]; then
        NAME="${NAME}_concat"
    else
        NAME="${NAME}_cvae_kl${KL}"
    fi

    local LOG="${LOG_DIR}/${NAME}.txt"
    local CKPT="${CKPT_DIR}/${NAME}.pt"

    # Build flags
    local FLAGS="--mode $MODE --dataset mosei --datapath $DATA --use_pt"
    FLAGS="$FLAGS --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS"
    FLAGS="$FLAGS --dropout_prob 0.2 --lr 0.001 --recon_weight 1.0"
    FLAGS="$FLAGS --seed $SEED --log_interval 500"
    FLAGS="$FLAGS --subset_size $SIZE"
    FLAGS="$FLAGS --name $CKPT"

    if [ "$MODE" = "cvae" ]; then
        FLAGS="$FLAGS --kl_weight $KL"
    fi

    # Build record config
    local RCONFIG="--config=mode=$MODE --config=subset_size=$SIZE --config=seed=$SEED"
    RCONFIG="$RCONFIG --config=lr=0.001 --config=batch_size=$BS --config=epochs=$EPOCHS"
    if [ "$MODE" = "cvae" ]; then
        RCONFIG="$RCONFIG --config=kl_weight=$KL --config=recon_weight=1.0"
    fi

    local CMD="echo '[$COUNT/12] $(date +%H:%M:%S) ${NAME}  # ${DESC}' && \
         PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py $FLAGS \
           > $LOG 2>&1 && \
         /usr/bin/python3 record_results.py $LOG $NAME $RCONFIG"
    CMDS+=("$CMD")
}

# 4 sizes × 3 configs
echo "── Subsampling sweep ──"
for SIZE in 1300 2500 5000 10000; do
    # Baseline: concat (no reconstruction, just zero-fill)
    run_one $SIZE 0 concat "concat @ ${SIZE}"

    # CVAE at two KL values spanning the optimal range
    run_one $SIZE 0.4 cvae "CVAE KL=0.4 @ ${SIZE}"
    run_one $SIZE 0.8 cvae "CVAE KL=0.8 @ ${SIZE}"
done

TOTAL=${#CMDS[@]}
echo ""
echo "Total: $TOTAL runs"
echo "Estimated: 1.3K ~12min | 2.5K ~25min | 5K ~50min | 10K ~1.5h each"
echo "6 parallel → ~2.5h wall-clock"
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
echo "SUBSAMPLE SWEEP ALL DONE at $(date)"
echo "Results: $LOG_DIR/"
echo "============================================================"
