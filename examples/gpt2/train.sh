TOTAL_UPDATES=150000    # Total number of training steps
WARMUP_UPDATES=500   # Warmup the learning rate over this many updates
PEAK_LR=0.0002          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=1024   # Max sequence length
MAX_POSITIONS=1024       # Num. positional embeddings (usually same as above)        # Number of sequences per batch (batch size)
UPDATE_FREQ=8          # Increase the batch size x
DATA_DIR=/mnt/lustre/share_data/qinzhen/nlp_proc/gptdata_bin
BATCH_SIZE=1
spring.submit run --gpu -n$1 --ntasks-per-node 8 --cpus-per-task 4 \
--job-name=gpt2 \
"fairseq-train --fp16 $DATA_DIR \
  --ddp-backend=fully_sharded --cpu-offload --distributed-world-size $1 --distributed-port 12343 \
    --task lm_gpt2 --criterion clm --checkpoint-activations \
    --arch transformer_lm_gpt2_big --tokens-per-sample $TOKENS_PER_SAMPLE \
    --tensorboard-logdir log \
    --optimizer cpu_adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --clip-norm 1.0 \
    --lr-scheduler cosine --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES \
    --weight-decay 0.1 \
    --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format json --log-interval 10 2>&1 | tee log.train"
