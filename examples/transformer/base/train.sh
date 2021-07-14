BATCH_SIZE=1
DATA_DIR=data-bin/wikitext-103
spring.submit arun \
    --gpu \
    -n$1 \
    --job-name=rua_fairseq_wiki103\
    --ntasks-per-node 8 --cpus-per-task 4 \
    "fairseq-train --task language_modeling \
    $DATA_DIR \
    --save-dir checkpoints/transformer_wikitext103 \
    --distributed-world-size $1  --distributed-port 12343\
    --arch transformer_lm_wiki103 \
    --max-update 286000 --lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp 
    --batch-size $BATCH_SIZE
    --log-interval 10 2>&1 | tee log.train"
