#!/usr/bin/env python

import os
from collections import defaultdict

import numpy as np

import config


RESULTS_DIR = os.path.join(config.MAIN_FOLDER, 'results')
EMOS = config.EMOS

print('')
print('#' * 12)
print('SemEval 2007')
print('#' * 12)
print('')

models = os.listdir(RESULTS_DIR)
for model in models:
    model_dir = os.path.join(RESULTS_DIR, model)
    collected_r = defaultdict(list)
    if 'gp' in model:
        collected_nlpd = defaultdict(list)
    folds = os.listdir(model_dir)
    for fold in folds:
        fold_dir = os.path.join(model_dir, fold)
        for emo in EMOS:
            metrics = np.loadtxt(os.path.join(fold_dir, emo + '.metrics'))
            collected_r[emo].append(metrics[0])
            if 'gp' in model:
                collected_nlpd[emo].append(metrics[2])
        #if fold == '6': break
    print(model)
    for emo in EMOS:
        if 'gp' in model:
            print("%10s\t%.3f\t%.2f" % (emo, np.mean(collected_r[emo]), np.mean(collected_nlpd[emo])))
        else:
            print("%10s\t%.3f" % (emo, np.mean(collected_r[emo])))
    avg_r = np.mean([np.mean(collected_r[emo]) for emo in EMOS])
    if 'gp' in model:
        avg_nlpd = np.mean([np.mean(collected_nlpd[emo]) for emo in EMOS])
        print("%10s\t%.3f\t%.2f" % ('AVERAGE', avg_r, avg_nlpd))
    else:
        print("%10s\t%.3f" % ('AVERAGE', avg_r))

# WASSA

EMOS.pop('disgust')
EMOS.pop('surprise')
print('')
print('#' * 10)
print('WASSA 2017')
print('#' * 10)
print('')
RESULTS_DIR = os.path.join(config.MAIN_FOLDER, 'wassa_results')
models = os.listdir(RESULTS_DIR)

for model in models:
    model_dir = os.path.join(RESULTS_DIR, model)
    metrics = {}
    for emo in EMOS:
        metrics[emo] = np.loadtxt(os.path.join(model_dir, emo, 'metrics'))
    print(model)
    for emo in EMOS:
        if 'gp' in model:
            print("%10s\t%.3f\t%.3f" % (emo, metrics[emo][0], metrics[emo][2]))
        else:
            print("%10s\t%.3f" % (emo, metrics[emo][0]))
    avg_r = np.mean([metrics[emo][0] for emo in EMOS])
    if 'gp' in model:
        avg_nlpd = np.mean([metrics[emo][2] for emo in EMOS])
        print("%10s\t%.3f\t%.3f" % ('AVERAGE', avg_r, avg_nlpd))
    else:
        print("%10s\t%.3f" % ('AVERAGE', avg_r))

