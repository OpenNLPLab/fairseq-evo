batch_size=1
spring.submit arun --gpu \
    "fairseq-eval-lm \
        data-bin/wikitext-103 \
        --path checkpoints/transformer_wikitext-103/checkpoint_best.pt \
        --batch-size 2
        --tokens-per-sample 512 \
        --context-window 400 2>&1 | tee eval.log"