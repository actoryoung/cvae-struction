#!/bin/bash
# ============================================================
# CVAE Hyperparameter Grid Sweep (mmap-optimized)
# 192 combinations: KL × Recon × Dropout × LR
# 6-way parallel, ~10 hours total
# Uses preprocessed .pt files with memory-mapped loading.
# ============================================================
set +e

PROJ_DIR="/home/ly/stu_work/projects/missing-modality-msa"
cd "$PROJ_DIR"

DATA="casp_dataset/mosei_pt"
EPOCHS=30
BS=32
WORKERS=0  # 0=main process loads data (avoids multiproc mem copy)
SEED=666

N_PARALLEL=6
USE_PT="--use_pt"

KL_WEIGHTS=(0.05 0.1 0.2 0.5)
RECON_WEIGHTS=(0.5 1.0 2.0 5.0)
DROPOUT_PROBS=(0.1 0.2 0.3)
LRS=(0.0008 0.001 0.002)

TOTAL=$(( ${#KL_WEIGHTS[@]} * ${#RECON_WEIGHTS[@]} * ${#DROPOUT_PROBS[@]} * ${#LRS[@]} ))
echo "============================================================"
echo "CVAE Grid Sweep: $TOTAL total combinations, $N_PARALLEL parallel"
echo "Data: $DATA (mmap .pt files)"
echo "KL:      ${KL_WEIGHTS[*]}"
echo "Recon:   ${RECON_WEIGHTS[*]}"
echo "Dropout: ${DROPOUT_PROBS[*]}"
echo "LR:      ${LRS[*]}"
echo "============================================================"
echo "Started at: $(date)"
echo ""

# Clean previous partial results
rm -f /tmp/sweep_grid/*.txt

# Build run list
declare -a CMDS
COUNT=0
for KL in "${KL_WEIGHTS[@]}"; do
  for RW in "${RECON_WEIGHTS[@]}"; do
    for DP in "${DROPOUT_PROBS[@]}"; do
      for LR in "${LRS[@]}"; do
        COUNT=$((COUNT + 1))
        NAME="kl${KL}_rw${RW}_dp${DP}_lr${LR}"
        LOG="/tmp/sweep_grid/${NAME}.txt"
        CKPT="checkpoints/sweep/${NAME}.pt"

        CMD="echo '[$COUNT/$TOTAL] $(date +%H:%M:%S) $NAME' && \
             PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py \
               --mode cvae --dataset mosei --datapath $DATA \
               $USE_PT \
               --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS \
               --kl_weight $KL --recon_weight $RW --dropout_prob $DP --lr $LR \
               --seed $SEED \
               --name $CKPT --log_interval 500 \
               2>&1 | tee $LOG && \
             /usr/bin/python3 record_results.py $LOG $NAME \
               --config=kl_weight=$KL --config=recon_weight=$RW \
               --config=dropout_prob=$DP --config=lr=$LR \
               --config=mode=cvae --config=batch_size=$BS --config=epochs=$EPOCHS"
        CMDS+=("$CMD")
      done
    done
  done
done

TOTAL_CMDS=${#CMDS[@]}
echo "Launching $TOTAL_CMDS jobs..."
echo ""

RUNNING=0
for i in "${!CMDS[@]}"; do
  # Wait if at capacity
  if [ $RUNNING -ge $N_PARALLEL ]; then
    wait -n
    RUNNING=$((RUNNING - 1))
  fi

  # Launch job in background
  eval "${CMDS[$i]}" &
  RUNNING=$((RUNNING + 1))
done

# Wait for remaining
wait

echo ""
echo "============================================================"
echo "ALL $TOTAL JOBS COMPLETED"
echo "Finished at: $(date)"
echo "Logs:    /tmp/sweep_grid/"
echo "Models:  checkpoints/sweep/"
echo "Results: results/sweep/"
echo "============================================================"
echo ""
echo "Generate summary:  PYTHONUNBUFFERED=1 python3 summarize_sweep.py"
