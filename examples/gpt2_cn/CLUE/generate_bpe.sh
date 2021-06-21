SPLITS="train dev test"
GLUE_DATA_FOLDER=$1
TASK=$2 # 
TASK_DATA_FOLDER="$GLUE_DATA_FOLDER/$TASK"
#TASKS="QQP MNLI QNLI MRPC RTE STS-B SST-2 CoLA"
if [ "$TASK" = "afqmc" ]
then
  INPUT_COUNT=2
elif [ "$TASK" = "iflytek" ]
then
  INPUT_COUNT=1
elif [ "$TASK" = "tnews" ]
then
  INPUT_COUNT=1
elif [ "$TASK" = "cmnli" ]
then
  INPUT_COUNT=2
elif [ "$TASK" = "csl" ]
then
  INPUT_COUNT=2
elif [ "$TASK" = "wsc" ]
then
  INPUT_COUNT=2
fi
for SPLIT in $SPLITS
do
  for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
  do
      LAN="input$INPUT_TYPE"
      echo "BPE encoding $SPLIT/$LAN"
      spring.submit run --cpus-per-task 4 \
      "python multiprocessing_bpe_encoder.py \
      --vocab gpt2_bpe/vocab.vocab \
      --vocab_model gpt2_bpe/vocab.model \
      --inputs "$TASK_DATA_FOLDER/processed/$SPLIT.raw.$LAN" \
      --outputs "$TASK_DATA_FOLDER/processed/$SPLIT.$LAN" \
      --workers 60 \
      --keep-empty"
  done
done
