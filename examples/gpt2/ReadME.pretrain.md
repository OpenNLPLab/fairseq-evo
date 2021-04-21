sh generate_bpe_token.sh #bpe 预处理
sh generate_bin_file.sh #可以先给个空dict.txt，实际处理不会用到
sh train.sh $gpus #数据路径中一定要有encoder.json 和 vocab.bpe
#wget -O $data/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
#wget -O $data/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe