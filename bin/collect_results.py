#!/usr/bin/env python

import os
from collections import defaultdict

import numpy as np

import config


RESULTS_DIR = os.path.join(config.MAIN_FOLDER, 'results')
EMOS = config.EMOS

models = os.listdir(RESULTS_DIR)
for model in models:
    model_dir = os.path.join(RESULTS_DIR, model)
    collected_r = defaultdict(list)
    folds = os.listdir(model_dir)
    for fold in folds:
        fold_dir = os.path.join(model_dir, fold)
        for emo in EMOS:
            metrics = np.loadtxt(os.path.join(fold_dir, emo + '.metrics'))
            collected_r[emo].append(metrics[0])
    print(model)
    for emo in EMOS:
        print("%10s\t%.4f" % (emo, np.mean(collected_r[emo])))
