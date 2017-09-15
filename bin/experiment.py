
import os
from collections import namedtuple

import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.stats.stats import pearsonr

import config


Data = namedtuple('Data', ['X_train',
                           'Y_train',
                           'X_test',
                           'Y_test'])


class Experiment(object):

    def __init__(self, model, data):
        self.model_name = model
        self.data = data

    def train_models(self):
        self.models = {}
        X_train = self.data.X_train
        for emo in config.EMOS:
            Y_train = self.data.Y_train[:, config.EMOS[emo]]
            model = self._train_model(X_train, Y_train)
            self.models[emo] = model

    def eval_models(self):
        self.predictions = {}
        self.metrics = {}
        for emo in config.EMOS:
            preds = self.models[emo].predict(self.data.X_test)
            self.predictions[emo] = preds
            self.metrics[emo] = pearsonr(preds, self.data.Y_test[:, config.EMOS[emo]])

    def save_metrics(self, results_dir):
        for emo in config.EMOS:
            np.savetxt(os.path.join(results_dir, emo + '.metrics'), self.metrics[emo], fmt='%.4f')

    def save_predictions(self, results_dir):
        for emo in config.EMOS:
            np.savetxt(os.path.join(results_dir, emo + '.preds'), self.predictions[emo], fmt='%.8f')
            
    def _train_model(self, X_train, Y_train):
        if self.model_name == 'ridge':
            return self._train_ridge(X_train, Y_train)

    def _train_ridge(self, X_train, Y_train):
        model = RidgeCV(alphas=np.logspace(-2, 2, 5))
        model.fit(X_train, Y_train)
        return model


