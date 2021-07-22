batch_size=1
data_dir=/mnt/lustre/qinzhen/nlpwork/dataset/data-bin/wikitext-103
ckpt=/mnt/lustre/qinzhen/nlpwork/transformer/experiment/checkpoints/sparse_transformer_lm_wiki103/checkpoint_best.pt
spring.submit arun --gpu \
    "fairseq-eval-lm \
        $data_dir \
        --path $ckpt \
        --batch-size $batch_size
        --tokens-per-sample 512 \
        --context-window 400 2>&1 | tee eval.log"