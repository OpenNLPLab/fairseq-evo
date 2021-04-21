TOTAL_UPDATES=250000    # Total number of training steps
WARMUP_UPDATES=20000   # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=1024   # Max sequence length
MAX_POSITIONS=1024       # Num. positional embeddings (usually same as above)        # Number of sequences per batch (batch size)
UPDATE_FREQ=1          # Increase the batch size 16x
DATA_DIR=/mnt/lustre/share_data/qinzhen/nlp_proc/gptdata_bin
BATCH_SIZE=1
spring.submit run --gpu -n$1 --ntasks-per-node 8 \
--job-name=debug \
"fairseq-train --fp16 $DATA_DIR \
  --distributed-world-size $1 --distributed-port 12343  \
    --task lm_gpt2 --criterion clm \
    --arch hf_gpt2 --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES \
    --weight-decay 0.01 \
    --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ --total-num-update $TOTAL_UPDATES\
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1"
