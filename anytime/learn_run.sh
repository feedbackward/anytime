#!/bin/bash

ALGO_ANCILLARY="SGD"
ALGO_MAIN="Ave"
BATCH="8"
DATA="adult"
ENTROPY="256117190779556056928268872043329970341"
LOSS="logistic"
MODEL="linreg_multi"
EPOCHS="30"
TASK="default"
TRIALS="10"
STEP="1.0"

python "learn_driver.py" --algo-ancillary="$ALGO_ANCILLARY" --algo-main="$ALGO_MAIN" --batch-size="$BATCH" --data="$DATA" --entropy="$ENTROPY" --loss="$LOSS" --model="$MODEL" --num-epochs="$EPOCHS" --num-trials="$TRIALS" --step-size="$STEP" --task-name="$TASK"

