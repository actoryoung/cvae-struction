#!/bin/bash
# Fill-in: MC + Contrastive @ KL=0.4 and KL=0.8
# 6 runs, testing whether strategies work at new KL peaks
set +e

PROJ_DIR="/home/ly/stu_work/projects/missing-modality-msa"
cd "$PROJ_DIR"

DATA="casp_dataset/mosei_pt"
EPOCHS=30; BS=32; WORKERS=0; SEED=666; N_PARALLEL=6
LOG_DIR="/tmp/combo_fillin"
CKPT_DIR="checkpoints/combo"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================================"
echo "Fill-in: MC + Contrastive @ KL=0.4 and KL=0.8 (6 runs)"
echo "============================================================"
echo "Started at: $(date)"

declare -a CMDS
COUNT=0

run_one() {
    local KL=$1 RW=$2 LR=$3 MC=$4 CW=$5 DESC=$6
    COUNT=$((COUNT + 1))
    local NAME="fillin_kl${KL}_rw${RW}_lr${LR}"
    [ "$MC" -gt 1 ] && NAME="${NAME}_mc${MC}"
    [ "$CW" != "0" ] && NAME="${NAME}_ct${CW}"
    local LOG="${LOG_DIR}/${NAME}.txt"
    local CKPT="${CKPT_DIR}/${NAME}.pt"

    local FLAGS="--mode cvae --dataset mosei --datapath $DATA --use_pt"
    FLAGS="$FLAGS --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS"
    FLAGS="$FLAGS --kl_weight $KL --recon_weight $RW --dropout_prob 0.2 --lr $LR"
    FLAGS="$FLAGS --seed $SEED --log_interval 500"
    FLAGS="$FLAGS --mc_samples $MC"
    [ "$CW" != "0" ] && FLAGS="$FLAGS --contrastive_weight $CW"
    FLAGS="$FLAGS --name $CKPT"

    local RCONFIG="--config=kl_weight=$KL --config=recon_weight=$RW --config=dropout_prob=0.2 --config=lr=$LR --config=mc_samples=$MC"
    [ "$CW" != "0" ] && RCONFIG="$RCONFIG --config=contrastive_weight=$CW"
    RCONFIG="$RCONFIG --config=mode=cvae --config=batch_size=$BS --config=epochs=$EPOCHS"

    local CMD="echo '[$COUNT/6] \$(date +%H:%M:%S) ${NAME}  # ${DESC}' && \
         PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py $FLAGS \
           > $LOG 2>&1 && \
         /usr/bin/python3 record_results.py $LOG $NAME $RCONFIG"
    CMDS+=("$CMD")
}

echo "KL=0.4 + MC (standard lr and best lr)"
run_one 0.4 1.0 0.001 5 0 "KL0.4+MC rw=1.0 lr=0.001"
run_one 0.4 0.5 0.002 5 0 "KL0.4+MC rw=0.5 lr=0.002"

echo "KL=0.4 + Contrastive (best cw=0.7, plus cw=0.5)"
run_one 0.4 1.0 0.001 1 0.7 "KL0.4+CT cw=0.7"
run_one 0.4 1.0 0.001 1 0.5 "KL0.4+CT cw=0.5"

echo "KL=0.8 + MC"
run_one 0.8 1.0 0.001 5 0 "KL0.8+MC rw=1.0"

echo "KL=0.8 + Contrastive"
run_one 0.8 1.0 0.001 1 0.7 "KL0.8+CT cw=0.7"

TOTAL=${#CMDS[@]}
echo ""
echo "Total: $TOTAL combinations"
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
echo "ALL $TOTAL FILL-IN JOBS COMPLETED at $(date)"
echo "============================================================"
