#!/bin/bash

MODEL=$1
DIR=`dirname $0`
MAIN_FOLDER=$($DIR/config.py)

for FOLD in $(seq 0 9);
do
    OMP_NUM_THREADS=1 $MAIN_FOLDER/bin/run.py --model $MODEL --folds $FOLD &
done
