#!/bin/bash
# ============================================================
# Multi-Seed Validation Experiments
# 2 extra seeds (42, 123) for main result table configs
# Seed=666 already completed in previous sweeps
# MOSEI: 4 configs × 2 seeds = 8 runs (~3h, 6 parallel)
# MOSI:  4 configs × 2 seeds = 8 runs (~30min, 4 parallel)
# ============================================================
set +e

PROJ_DIR="/home/ly/stu_work/projects/missing-modality-msa"
cd "$PROJ_DIR"

MOSI_DATA="casp_dataset/mosi.pkl"
MOSEI_DATA="casp_dataset/mosei_pt"
EPOCHS=30; BS=32; WORKERS=0; N_PARALLEL=6
LOG_DIR="/tmp/multi_seed"
mkdir -p "$LOG_DIR" "checkpoints/multi_seed"

echo "============================================================"
echo "Multi-Seed Validation: 2 extra seeds (42, 123)"
echo "MOSEI: 8 runs | MOSI: 8 runs | Total: 16"
echo "============================================================"
echo "Started at: $(date)"
echo ""

declare -a CMDS
COUNT=0

run_one() {
    local DATASET=$1 KL=$2 RW=$3 CW=$4 MC=$5 SEED=$6 MODE=$7 DESC=$8
    local DATA_PATH=$9
    COUNT=$((COUNT + 1))

    # Build name
    local NAME="seed${SEED}_${DATASET}"
    [ "$MODE" = "concat" ] && NAME="${NAME}_concat"
    [ "$MODE" = "cvae" ] && NAME="${NAME}_kl${KL}"
    [ "$CW" != "0" ] && NAME="${NAME}_ct${CW}"

    local LOG="${LOG_DIR}/${NAME}.txt"
    local CKPT="checkpoints/multi_seed/${NAME}.pt"

    # Build flags
    local FLAGS="--mode $MODE --dataset $DATASET --datapath $DATA_PATH"
    if [ "$DATASET" = "mosei" ]; then
        FLAGS="$FLAGS --use_pt"
    fi
    FLAGS="$FLAGS --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS"
    FLAGS="$FLAGS --dropout_prob 0.2 --lr 0.001"
    FLAGS="$FLAGS --seed $SEED --log_interval 500"

    if [ "$MODE" = "cvae" ]; then
        FLAGS="$FLAGS --kl_weight $KL --recon_weight $RW --mc_samples $MC"
        if [ "$CW" != "0" ]; then
            FLAGS="$FLAGS --contrastive_weight $CW"
        fi
    fi
    FLAGS="$FLAGS --name $CKPT"

    # Build record config
    local RCONFIG="--config=dataset=$DATASET --config=mode=$MODE --config=seed=$SEED"
    [ "$MODE" = "cvae" ] && RCONFIG="$RCONFIG --config=kl_weight=$KL --config=recon_weight=$RW"
    [ "$CW" != "0" ] && RCONFIG="$RCONFIG --config=contrastive_weight=$CW"
    RCONFIG="$RCONFIG --config=lr=0.001 --config=batch_size=$BS --config=epochs=$EPOCHS"

    local CMD="echo '[$COUNT/16] \$(date +%H:%M:%S) ${NAME}  # ${DESC}' && \
         PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py $FLAGS \
           > $LOG 2>&1 && \
         /usr/bin/python3 record_results.py $LOG $NAME $RCONFIG"
    CMDS+=("$CMD")
}

echo "── MOSEI (8 runs) ──"
for SEED in 20260113 20040169; do
    # concat baseline
    run_one mosei 0 0 0 1 $SEED concat "MOSEI concat" "$MOSEI_DATA"
    # KL=0.4 pure CVAE
    run_one mosei 0.4 1.0 0 1 $SEED cvae "MOSEI KL=0.4" "$MOSEI_DATA"
    # KL=0.4 + Contrastive cw=0.7 (overall best)
    run_one mosei 0.4 1.0 0.7 1 $SEED cvae "MOSEI KL=0.4+CT0.7" "$MOSEI_DATA"
    # KL=0.8 pure CVAE (best pure)
    run_one mosei 0.8 1.0 0 1 $SEED cvae "MOSEI KL=0.8" "$MOSEI_DATA"
done

echo "── MOSI (8 runs) ──"
for SEED in 20260113 20040169; do
    # concat baseline
    run_one mosi 0 0 0 1 $SEED concat "MOSI concat" "$MOSI_DATA"
    # KL=0.001 CVAE (best MissT on MOSI)
    run_one mosi 0.001 1.0 0 1 $SEED cvae "MOSI KL=0.001" "$MOSI_DATA"
    # KL=0.4 CVAE (MOSEI best KL for comparison)
    run_one mosi 0.4 1.0 0 1 $SEED cvae "MOSI KL=0.4" "$MOSI_DATA"
    # KL=0.8 CVAE (high KL comparison)
    run_one mosi 0.8 1.0 0 1 $SEED cvae "MOSI KL=0.8" "$MOSI_DATA"
done

TOTAL=${#CMDS[@]}
echo ""
echo "Total: $TOTAL combinations (16 runs)"
echo "Estimated: MOSEI ~3h + MOSI ~30min, 6 parallel"
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
echo "MULTI-SEED ALL DONE at $(date)"
echo "Aggregate: python3 aggregate_seeds.py"
echo "============================================================"
