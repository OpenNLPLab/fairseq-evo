SPLITS="train dev test"
GLUE_DATA_FOLDER=$1
TASK=$2 #
TASK_DATA_FOLDER="$GLUE_DATA_FOLDER/$TASK"
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
for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
  do
    LAN="input$INPUT_TYPE"
    spring.submit run --cpus-per-task 4 \
    "fairseq-preprocess \
      --only-source \
      --trainpref "$TASK_DATA_FOLDER/processed/train.$LAN" \
      --validpref "$TASK_DATA_FOLDER/processed/dev.$LAN" \
      --testpref "$TASK_DATA_FOLDER/processed/test.$LAN" \
      --destdir "$TASK-bin/$LAN" \
      --not_append_eos \
      --already_numberized \
      --workers 60 \
      --srcdict dict.txt"
  done
  if [[ "$TASK" !=  "STS-B" ]]
  then
    spring.submit run --cpus-per-task 4 \
    "fairseq-preprocess \
      --only-source \
      --trainpref "$TASK_DATA_FOLDER/processed/train.label" \
      --validpref "$TASK_DATA_FOLDER/processed/dev.label" \
      --destdir "$TASK-bin/label" \
      --workers 60;"
  fi
