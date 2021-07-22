TOTAL_UPDATES=1500000    # Total number of training steps
WARMUP_UPDATES=500   # Warmup the learning rate over this many updates
PEAK_LR=0.00000002          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=3072   # Max sequence length
MAX_TOKEN=3072          # 最终token长度=min(TOKENS_PER_SAMPLE, MAX_TOKEN)
MAX_POSITIONS=3072       # Num. positional embeddings (usually same as above)        # Number of sequences per batch (batch size)
UPDATE_FREQ=8          # Increase the batch size x
DATA_DIR=/mnt/lustre/qinzhen/nlpwork/dataset/data-bin/wikitext-103
BATCH_SIZE=1
ARCH=performer_lm_small_wiki103
#ARCH=transformer_lm_rfa_small_wiki103
#ARCH=transformer_lm_small_wiki103
# JOB_NAME=$ARCH

spring.submit arun --gpu -n$1 --ntasks-per-node 8 --cpus-per-task 4 \
--job-name=$ARCH \
"fairseq-train --task language_modeling \
    $DATA_DIR \
    --ddp-backend=legacy_ddp \
    --criterion adaptive_loss \
    --save-dir checkpoints/$ARCH \
    --arch $ARCH \
    --max-tokens $MAX_TOKEN \
    --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06  --clip-norm 0.1 \
    --lr-scheduler cosine --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES \
    --weight-decay 0.1 \
    --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --max-update $TOTAL_UPDATES"