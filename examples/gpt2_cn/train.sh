TOTAL_UPDATES=300000    # Total number of training steps
WARMUP_UPDATES=500   # Warmup the learning rate over this many updates
PEAK_LR=0.0002          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=1024   # Max sequence length
MAX_POSITIONS=1024       # Num. positional embeddings (usually same as above)        # Number of sequences per batch (batch size)
UPDATE_FREQ=1          # Increase the batch size x
DATA_DIR=./data-bin/train_1
BATCH_SIZE=10
spring.submit run  --gpu -n$1 -x SH-IDC1-10-198-34-[69] --ntasks-per-node 8 --cpus-per-task 4 \
--job-name=gpt_small_cn \
"fairseq-train --fp16 $DATA_DIR \
    --distributed-world-size $1 --distributed-port 12343 \
    --task lm_gpt2_cn --criterion clm \
    --arch transformer_lm_gpt --tokens-per-sample $TOKENS_PER_SAMPLE \
    --tensorboard-logdir log  \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --clip-norm 1.0 \
    --lr-scheduler cosine --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES \
    --weight-decay 0.1 \
    --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format json --log-interval 10"
