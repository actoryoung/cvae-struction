#!/bin/bash
# ============================================================
# Combo Sweep v2: 叠加有效策略 (跳过已完成的 6 个)
# ✅ Block 0:  Verification baselines (2) — COMPLETED
# ✅ Block 0.5: KL=0.3/0.4/0.6/0.8 — COMPLETED
# 🔄 Block 0.5: KL=1.0 + Block 1-4: 15 个剩余
# ============================================================
set +e

PROJ_DIR="/home/ly/stu_work/projects/missing-modality-msa"
cd "$PROJ_DIR"

DATA="casp_dataset/mosei_pt"
EPOCHS=30; BS=32; WORKERS=0; SEED=666; N_PARALLEL=6
LOG_DIR="/tmp/combo_sweep"
CKPT_DIR="checkpoints/combo"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================================"
echo "Combo Sweep v2: 15 remaining (6 completed in v1)"
echo "============================================================"
echo "Started at: $(date)"
echo ""

declare -a CMDS
COUNT=0

run_one() {
    local KL=$1 RW=$2 DP=$3 LR=$4 MC=$5 CW=$6 LATENT=$7 HIDDEN=$8 DESC=$9
    COUNT=$((COUNT + 1))
    local NAME="combo_kl${KL}_rw${RW}_dp${DP}_lr${LR}"
    [ "$MC" -gt 1 ] && NAME="${NAME}_mc${MC}"
    [ "$CW" != "0" ] && NAME="${NAME}_ct${CW}"
    [ "$LATENT" != "32" ] && NAME="${NAME}_lat${LATENT}"
    [ "$HIDDEN" != "64" ] && NAME="${NAME}_hid${HIDDEN}"
    local LOG="${LOG_DIR}/${NAME}.txt"
    local CKPT="${CKPT_DIR}/${NAME}.pt"

    # Build flags
    local FLAGS="--mode cvae --dataset mosei --datapath $DATA --use_pt"
    FLAGS="$FLAGS --num_epochs $EPOCHS --batch_size $BS --num_workers $WORKERS"
    FLAGS="$FLAGS --kl_weight $KL --recon_weight $RW --dropout_prob $DP --lr $LR"
    FLAGS="$FLAGS --seed $SEED --log_interval 500"
    FLAGS="$FLAGS --mc_samples $MC"
    [ "$CW" != "0" ] && FLAGS="$FLAGS --contrastive_weight $CW"
    [ "$LATENT" != "32" ] && FLAGS="$FLAGS --cvae_latent $LATENT"
    [ "$HIDDEN" != "64" ] && FLAGS="$FLAGS --cvae_hidden $HIDDEN"
    FLAGS="$FLAGS --name $CKPT"

    # Build record config
    local RCONFIG="--config=kl_weight=$KL --config=recon_weight=$RW"
    RCONFIG="$RCONFIG --config=dropout_prob=$DP --config=lr=$LR"
    RCONFIG="$RCONFIG --config=mc_samples=$MC"
    [ "$CW" != "0" ] && RCONFIG="$RCONFIG --config=contrastive_weight=$CW"
    [ "$LATENT" != "32" ] && RCONFIG="$RCONFIG --config=cvae_latent=$LATENT"
    [ "$HIDDEN" != "64" ] && RCONFIG="$RCONFIG --config=cvae_hidden=$HIDDEN"
    RCONFIG="$RCONFIG --config=mode=cvae --config=batch_size=$BS --config=epochs=$EPOCHS"

    local CMD="echo '[$COUNT/15] \$(date +%H:%M:%S) ${NAME}  # ${DESC}' && \
         PYTHONUNBUFFERED=1 /usr/bin/python3 train_cvae.py $FLAGS \
           > $LOG 2>&1 && \
         /usr/bin/python3 record_results.py $LOG $NAME $RCONFIG"
    CMDS+=("$CMD")
}

# ══════════════════════════════════════════════════════════
# Block 0.5 (remaining): KL=1.0 (1 run)
# ══════════════════════════════════════════════════════════
echo "Block 0.5: KL Refinement (remaining: KL=1.0)"
run_one 1.0 1.0 0.2 0.001 1 0 32 64 "KL refinement: kl=1.0"

# ══════════════════════════════════════════════════════════
# Block 1: KL=0.5 + MC Inference (4 runs)
# ══════════════════════════════════════════════════════════
echo "Block 1: KL=0.5 + MC (P0-1)"
run_one 0.5 0.5 0.2 0.001 5 0 32 64 "KL0.5+MC5 rw=0.5"
run_one 0.5 1.0 0.2 0.001 5 0 32 64 "KL0.5+MC5 rw=1.0"
run_one 0.5 0.5 0.2 0.002 5 0 32 64 "KL0.5+MC5 rw=0.5 lr=0.002"
run_one 0.5 1.0 0.2 0.002 5 0 32 64 "KL0.5+MC5 rw=1.0 lr=0.002"

# ══════════════════════════════════════════════════════════
# Block 2: KL=0.5 + Contrastive (6 runs)
# ══════════════════════════════════════════════════════════
echo "Block 2: KL=0.5 + Contrastive (P0-2)"
for RW in 0.5 1.0; do
  for CW in 0.3 0.5 0.7; do
    run_one 0.5 $RW 0.2 0.001 1 $CW 32 64 "KL0.5+Contrast rw=$RW cw=$CW"
  done
done

# ══════════════════════════════════════════════════════════
# Block 3: KL=0.5 + Contrastive + MC (2 runs)
# ══════════════════════════════════════════════════════════
echo "Block 3: KL=0.5 + Contrastive + MC (P0-3)"
run_one 0.5 0.5 0.2 0.001 5 0.5 32 64 "KL0.5+Contrast0.5+MC5 rw=0.5"
run_one 0.5 1.0 0.2 0.001 5 0.5 32 64 "KL0.5+Contrast0.5+MC5 rw=1.0"

# ══════════════════════════════════════════════════════════
# Block 4: KL=0.5 + Capacity B + MC (2 runs)
# ══════════════════════════════════════════════════════════
echo "Block 4: KL=0.5 + Capacity B + MC (P1-1)"
run_one 0.5 0.5 0.2 0.001 5 0 64 256 "KL0.5+CapB+MC5 rw=0.5"
run_one 0.5 1.0 0.2 0.001 5 0 64 256 "KL0.5+CapB+MC5 rw=1.0"

TOTAL=${#CMDS[@]}
echo ""
echo "Total: $TOTAL combinations (15 remaining)"
echo "Previously completed: kl0.3, kl0.4, kl0.6, kl0.8, old-ct0.5, old-mc5"
echo ""

# ─── Launch (keep existing logs) ───
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
echo "ALL $TOTAL COMBO JOBS COMPLETED at $(date)"
echo "============================================================"
