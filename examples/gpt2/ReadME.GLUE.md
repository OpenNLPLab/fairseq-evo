# Download data
```bash
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all
```
### 2) Preprocess GLUE task data:
sh preprocess_GLUE_tasks.sh glue_data <glue_task_name>
`glue_task_name` is one of the following:
`{ALL, QQP, MNLI, QNLI, MRPC, RTE, STS-B, SST-2, CoLA}`
Use `ALL` for preprocessing all the glue tasks.
### 3) Fine-tuning on GLUE task:
sh train.sh $gpus $TASK_NAME
For each of the GLUE task, you will need to use following cmd-line arguments:

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-classes` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 1e-5 | 1e-5 | 1e-5 | 2e-5 | 1e-5 | 1e-5 | 1e-5 | 2e-5
`--batch-size` | 32 | 32 | 32 | 16 | 32 | 16 | 16 | 16
`--total-num-update` | 123873 | 33112 | 113272 | 2036 | 20935 | 2296 | 5336 | 3598
`--warmup-updates` | 7432 | 1986 | 28318 | 122 | 1256 | 137 | 320 | 214

For `STS-B` additionally add `--regression-target --best-checkpoint-metric loss` and remove `--maximize-best-checkpoint-metric`.

**Note:**

a) `--total-num-updates` is used by `--polynomial_decay` scheduler and is calculated for `--max-epoch=10` and `--batch-size=16/32` depending on the task.

b) Above cmd-args and hyperparams are tested on one Nvidia `V100` GPU with `32gb` of memory for each task. Depending on the GPU memory resources available to you, you can use increase `--update-freq` and reduce `--batch-size`.

c) All the settings in above table are suggested settings based on our hyperparam search within a fixed search space (for careful comparison across models). You might be able to find better metrics with wider hyperparam search.  
### Inference on GLUE task
python infer.py 
## infer.py 里面数据路径、模型路径需要修改 line18行不同TASK读取的格式不同，对应的sent1,sent2,label顺序需要调整