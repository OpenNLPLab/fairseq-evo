mkdir -p gpt2_bpe
for SPLIT in train ; do \
    spring.submit run --cpus-per-task 32 "python multiprocessing_bpe_encoder.py \
        --vocab gpt2_bpe/vocab.vocab \
        --vocab_model gpt2_bpe/vocab.model \
        --inputs corpus/${SPLIT}.txt \
        --outputs corpus/${SPLIT}.bpe \
        --keep-empty \
        --workers 30" ;\
done
