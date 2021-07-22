
BATCH_SIZE=1
DATA_DIR=/mnt/lustre/qinzhen/nlpwork/dataset/data-bin/wikitext-103
LR=0.05
MIN_LR=0.00001
Warmup=6000
MAX_UPDATES=150000
MAX_UPDATES=300000
ARCH=reformer_lm_wiki103

spring.submit arun \
    --gpu \
    -w SH-IDC1-10-198-34-34 \
    -n$1 \
    --job-name=$ARCH \
    --ntasks-per-node 8 --cpus-per-task 4 \
    "fairseq-train --task language_modeling \
    $DATA_DIR \
    --save-dir checkpoints/$ARCH \
    --distributed-world-size $1  --distributed-port 12343\
    --arch $ARCH \
    --max-update $MAX_UPDATES --lr $LR --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates $Warmup --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr $MIN_LR --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp 
    --batch-size $BATCH_SIZE
    --log-interval 10 2>&1 | tee log.train"
