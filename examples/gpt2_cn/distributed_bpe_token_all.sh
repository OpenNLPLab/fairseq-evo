mkdir -p gpt2_bpe
for SPLIT in train ; do \
  spring.submit run --cpus-per-task 8 -w SH-IDC1-10-198-34-46 -n8 --ntasks-per-node 8 "python distributed_bpe_encoder.py \
        --vocab gpt2_bpe/vocab.vocab \
        --vocab_model gpt2_bpe/vocab.model \
        --inputs corpus/ \
        --outputs corpus/${SPLIT}.bpe \
        --keep-empty \
        --writen_files_dir corpus/${SPLIT}_writen_files \
        --workers 8" ;\
done
