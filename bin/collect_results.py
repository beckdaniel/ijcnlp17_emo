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
        print("%10s\t%.4f" % (emo, np.mean(collected_r[emo])))
        #if 'gp' in model:
        #    print("%10s\t%.4f" % (emo, np.mean(collected_nlpd[emo])))
