BATCH_SIZE=1
DATA_DIR=/mnt/lustre/qinzhen/nlpwork/dataset/data-bin/wikitext-103
spring.submit arun \
    --gpu \
    -w SH-IDC1-10-198-34-34 \
    -n$1 \
    --job-name=rua_fairseq_wiki103\
    --ntasks-per-node 8 --cpus-per-task 4 \
    "fairseq-train --task language_modeling \
    $DATA_DIR \
    --save-dir checkpoints/transformer_wikitext103_base_test \
    --distributed-world-size $1  --distributed-port 12343\
    --arch transformer_lm_wiki103 \
    --max-update 150000 --lr 1.0 --lr-period-updates 60000 --lr-scheduler cosine \
    --warmup-updates 500 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
    --tensorboard-logdir log \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp 
    --batch-size $BATCH_SIZE
    --log-interval 10 2>&1 | tee log.train"
