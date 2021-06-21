spring.submit run --cpus-per-task 32 \
"fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref corpus/train.bpe \
    --destdir data-bin/new_2/ \
    --already_numberized \
    --not_append_eos \
    --workers 60"