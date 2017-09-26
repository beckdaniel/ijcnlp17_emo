#!/bin/bash

DIR=`dirname $0`
MAIN_FOLDER=$($DIR/config.py)
BIN=$MAIN_FOLDER/bin

# Perform SemEval 2007 experiments

$BIN/run_model.sh ridge
$BIN/run_model.sh svr
$BIN/run_model.sh gp_linear
$BIN/run_model.sh gp_rbf
$BIN/run_model.sh gp_mat32
$BIN/run_model.sh gp_mat52

# Perform WASSA 2017 experiments

$BIN/run_wassa.py ridge
$BIN/run_wassa.py gp_linear
$BIN/run_wassa.py gp_rbf
$BIN/run_wassa.py gp_mat32
$BIN/run_wassa.py gp_mat52
