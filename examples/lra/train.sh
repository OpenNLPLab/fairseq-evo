# Set up training envs. Same for all tasks.
seed=1

DATA=/data/qinzhen/data/lra
# mkdir -p ${SAVE}
# cp $0 ${SAVE}/run.sh
model=tnn
SAVE_ROOT=checkpoints


for task in listops imdb-4000 aan cifar10 pathfinder path-x
do
    # sleep 5
    SAVE=${SAVE_ROOT}/$task
    bash ${task}.sh ${seed} ${DATA}/${task} ${model} ${SAVE}
done