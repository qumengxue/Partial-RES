#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

PORT=${PORT:-29507}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    --use_env $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
