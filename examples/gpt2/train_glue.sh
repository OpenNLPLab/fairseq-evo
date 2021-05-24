TOTAL_NUM_UPDATES=5336  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=320      # 6 percent of the number of updates1
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=8        # Batch size.
MAX_POSITIONS=512
MODEL_PATH='checkpoints/checkpoint_best.pt'
DATA=$2
spring.submit run --gpu -n$1 --ntasks-per-node 8 \
"fairseq-train $DATA \
    --distributed-world-size $1 --distributed-port 12343
    --pretrained-model $MODEL_PATH
    --max-positions $MAX_POSITIONS
    --num-classes $NUM_CLASSES
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction_gpt2 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch hf_gpt2 \
    --num-labels $NUM_CLASSES
    --tie-word-embeddings False \
    --criterion sentence_prediction_gpt2 \
    --weight-decay 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;"
