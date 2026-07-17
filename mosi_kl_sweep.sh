#!/bin/bash
# MOSI KL sweep: test low KL range for small dataset
set +e

PROJ_DIR="/home/ly/stu_work/projects/missing-modality-msa"
cd "$PROJ_DIR"
DATA="casp_dataset/mosi.pkl"
EPOCHS=30; BS=32; SEED=666; N_PARALLEL=4
LOG_DIR="/tmp/mosi_kl"
mkdir -p "$LOG_DIR" "checkpoints/mosi"

echo "============================================================"
echo "MOSI KL Sweep: test KL ∈ [0.01, 0.05, 0.1, 0.2, 0.3] (5 runs)"
echo "============================================================"
echo "Started at: $(date)"

declare -a CMDS
COUNT=0

for KL in 0.01 0.05 0.1 0.2 0.3; do
    COUNT=$((COUNT + 1))
    NAME="mosi_kl${KL}"
    LOG="${LOG_DIR}/${NAME}.txt"
    CKPT="checkpoints/mosi/${NAME}.pt"
    
    CMD="echo '[$COUNT/5] \$(date +%H:%M:%S) MOSI KL=${KL}' && \
         PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py \
           --mode cvae --dataset mosi --datapath $DATA \
           --num_epochs 30 --batch_size $BS --num_workers 0 \
           --kl_weight $KL --recon_weight 1.0 --dropout_prob 0.2 --lr 0.001 \
           --seed $SEED --log_interval 50 \
           --name $CKPT \
           > $LOG 2>&1 && \
         /usr/bin/python3 record_results.py $LOG $NAME \
           --config=dataset=mosi --config=kl_weight=$KL --config=rw=1.0 --config=lr=0.001"
    CMDS+=("$CMD")
done

echo "Total: ${#CMDS[@]} jobs"

for i in "${!CMDS[@]}"; do
    eval "${CMDS[$i]}" &
done
wait

echo ""
echo "MOSI KL SWEEP DONE at $(date)"
