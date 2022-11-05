seed=$1
DATA=$2
model=$3
SAVE=$4

model=${model}_lra_cifar10

bs=50
bs=1

# fairseq-train ${DATA} \
#     --seed $seed --ddp-backend c10d --find-unused-parameters \
#     -a ${model} --task lra-image --input-type image --pixel-normalization 0.48 0.24 \
#     --encoder-layers 8 \
#     --activation-fn 'silu' --attention-activation-fn 'laplace' \
#     --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
#     --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#     --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
#     --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.02 \
#     --batch-size $bs --sentence-avg --update-freq 1 --max-update 180000 \
#     --lr-scheduler linear_decay --total-num-update 180000 --end-learning-rate 0.0 \
#     --warmup-updates 9000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 \
#     --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0

fairseq-train ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-image --input-type image --pixel-normalization 0.48 0.24 \
    --encoder-layers 8 \
    --activation-fn 'silu' --attention-activation-fn 'laplace' \
    --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.02 \
    --batch-size $bs --sentence-avg --update-freq 1 --max-update 2 \
    --lr-scheduler linear_decay --total-num-update 2 --end-learning-rate 0.0 \
    --warmup-updates 1 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 \
    --save-dir ${SAVE} --log-format simple --log-interval 1 --num-workers 0 \
    --max-positions 10 \
    --disable-validation \
    --no-save 2>&1 | tee ${model}.log