##1)download clue data
下载链接在 https://github.com/CLUEbenchmark/CLUE

### 2)数据预处理
python preprocess_clue.py --task $TASK --data_dir $DATA_DIR --output_dir $OUTPUT_DIR
##bash
sh generate_bpe.sh $GLUE_DATA_FOLDER $TASK
sh generate_bin.sh $GLUE_DATA_FOLDER $TASK

### 3) train 
sh train_clue.sh 8
#### 4) inference
python infer.py
