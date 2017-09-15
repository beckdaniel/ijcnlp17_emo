#!/usr/bin/env python

import argparse
import os
import subprocess
import numpy as np

import util
import config
import experiment


# Constants
MAIN = config.MAIN_FOLDER
EMBS = config.EMBS
EMOS = config.EMOS


# Parse input
parser = argparse.ArgumentParser(description='Train and test models')
parser.add_argument('-m', '--model', type=str, default='ridge')
parser.add_argument('-f', '--folds', nargs='*', type=int, default=[0])
args = parser.parse_args()


# Load embeddings
embs = util.load_embs(EMBS)


# For each cross-validation fold...
for fold in args.folds:
    # Load data
    folder = os.path.join(MAIN, 'splits', str(fold))
    with open(os.path.join(folder, 'instances.train.txt')) as f:
        X_train = [util.preprocess_sent(line.split('_')[1]) for line in f]
    with open(os.path.join(folder, 'instances.test.txt')) as f:
        X_test = [util.preprocess_sent(line.split('_')[1]) for line in f]
    Y_train = np.loadtxt(os.path.join(folder, 'emotion_scores.train.txt'))[:, 1:]
    Y_test = np.loadtxt(os.path.join(folder, 'emotion_scores.test.txt'))[:, 1:]

    # Preprocess sents
    X_train = np.array([util.average_sent(sent, embs) for sent in X_train])
    X_test = np.array([util.average_sent(sent, embs) for sent in X_test])

    # Train and evaluate model (on all emotions)
    data = experiment.Data(X_train, Y_train, X_test, Y_test)
    exp = experiment.Experiment(args.model, data)
    exp.train_models()
    exp.eval_models()

    # Log into results folder
    results_dir = os.path.join(MAIN, 'results', args.model, str(fold))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    exp.save_metrics(results_dir)
    exp.save_predictions(results_dir)
