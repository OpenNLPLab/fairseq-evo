TOTAL_NUM_UPDATES=343340  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=400     # 6 percent of the number of updates1
LR=3e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=4        # Batch size.
MAX_POSITIONS=1024
DATA=/mnt/lustre/liuzexiang/Code/fairseq/exp/cn_small_clue_test/clue_data/afqmc-bin/
MODEL='../../gpt2_cn_small/checkpoints/checkpoint_6_240000.pt'
spring.submit run --gpu -n$1 --ntasks-per-node 8 -x SH-IDC1-10-198-34-45 \
"fairseq-train $DATA \
    --max-positions $MAX_POSITIONS \
    --distributed-world-size $1  --distributed-port 12345 \
    --num-classes $NUM_CLASSES \
    --pretrained-model $MODEL
    --batch-size $MAX_SENTENCES \
    --task sentence_prediction_gpt2_cn \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch transformer_lm_gpt \
    --criterion sentence_prediction_gpt2 \
    --weight-decay 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 5 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
     --log-format json --log-interval 10 2>&1 | tee log.train"
