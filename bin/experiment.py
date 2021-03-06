
import os
from collections import namedtuple

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from scipy.stats.stats import pearsonr
import gpflow
import GPy

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
            if self.model_name == 'ridge' or self.model_name == 'svr':
                preds = self.models[emo].predict(self.data.X_test)
                self.predictions[emo] = preds
                self.metrics[emo] = pearsonr(preds, self.data.Y_test[:, config.EMOS[emo]])
            elif 'gp' in self.model_name:
                #preds = self.models[emo].predict_y(self.data.X_test)
                preds = self.models[emo].predict(self.data.X_test)
                self.predictions[emo] = preds[0]
                Y_test = self.data.Y_test[:, config.EMOS[emo]]
                metrics = list(pearsonr(preds[0].flatten(), Y_test))
                #nlpd = -np.mean(self.models[emo].predict_density(self.data.X_test, Y_test[:, None]))
                nlpd = -np.mean(self.models[emo].log_predictive_density(self.data.X_test, Y_test[:, None]))
                metrics.append(nlpd)
                self.metrics[emo] = metrics

    def save_metrics(self, results_dir):
        for emo in config.EMOS:
            np.savetxt(os.path.join(results_dir, emo + '.metrics'), self.metrics[emo], fmt='%.4f')

    def save_predictions(self, results_dir):
        for emo in config.EMOS:
            np.savetxt(os.path.join(results_dir, emo + '.preds'), self.predictions[emo], fmt='%.8f')
            
    def _train_model(self, X_train, Y_train):
        if self.model_name == 'ridge':
            return self._train_ridge(X_train, Y_train)
        elif self.model_name == 'svr':
            return self._train_svr(X_train, Y_train)
        elif 'gp' in self.model_name:
            kernel_name = self.model_name.split('_')[1]
            return self._train_gp(X_train, Y_train, kernel_name)

    def _train_ridge(self, X_train, Y_train):
        model = RidgeCV(alphas=np.logspace(-2, 2, 5))
        model.fit(X_train, Y_train)
        return model

    def _train_svr(self, X_train, Y_train):
        tuned_parameters = [{'C': np.logspace(-2, 2 ,5),
                             'gamma': np.logspace(-2, 2, 5),
                             'epsilon': np.logspace(-3, 1, 5)}]
        model = GridSearchCV(SVR(kernel='rbf'), tuned_parameters)
        model.fit(X_train, Y_train)
        return model

    def _train_gp(self, X_train, Y_train, kernel_name):
        input_dim = X_train.shape[1]
        Y_train = Y_train[:, None]
        if kernel_name == 'rbf':
            #kernel = gpflow.kernels.RBF(input_dim)
            kernel = GPy.kern.RBF(input_dim)
        elif kernel_name == 'mat32':
            #kernel = gpflow.kernels.Matern32(input_dim)
            kernel = GPy.kern.Matern32(input_dim)
        elif kernel_name == 'mat52':
            #kernel = gpflow.kernels.Matern52(input_dim)
            kernel = GPy.kern.Matern52(input_dim)
        elif kernel_name == 'linear':
            kernel = GPy.kern.Linear(input_dim)
        #kernel = kernel + gpflow.kernels.Bias(input_dim)
        kernel = kernel + GPy.kern.Bias(input_dim)
        #model = gpflow.gpr.GPR(X_train, Y_train, kern=kernel)
        model = GPy.models.GPRegression(X_train, Y_train, kernel=kernel)
        print(model)
        model.optimize(messages=True)
        print(model)
        return model


class WASSAExperiment(Experiment):

    def __init__(self, model, data):
        super(WASSAExperiment, self).__init__(model, data)
        
    def train_models(self):
        self.models = {}
        X_train = self.data.X_train
        Y_train = self.data.Y_train
        model = self._train_model(X_train, Y_train)
        self.model = model

    def eval_models(self):
        if self.model_name == 'ridge' or self.model_name == 'svr':
            preds = self.model.predict(self.data.X_test)
            self.predictions = preds
            self.metrics = pearsonr(preds, self.data.Y_test)
        elif 'gp' in self.model_name:
            #preds = self.models[emo].predict_y(self.data.X_test)
            preds = self.model.predict(self.data.X_test)
            self.predictions = preds[0]
            Y_test = self.data.Y_test
            metrics = list(pearsonr(preds[0].flatten(), Y_test))
            #nlpd = -np.mean(self.models[emo].predict_density(self.data.X_test, Y_test[:, None]))
            nlpd = -np.mean(self.model.log_predictive_density(self.data.X_test, Y_test[:, None]))
            metrics.append(nlpd)
            self.metrics = metrics

    def save_metrics(self, results_dir):
        np.savetxt(os.path.join(results_dir, 'metrics'), self.metrics, fmt='%.4f')

    def save_predictions(self, results_dir):
        np.savetxt(os.path.join(results_dir, 'preds'), self.predictions, fmt='%.8f')

    def _train_svr(self, X_train, Y_train):
        tuned_parameters = [{'C': np.logspace(-20, 1 ,7),
                             'gamma': np.logspace(-20, 1, 7),
                             'epsilon': np.logspace(-1, 3, 7)}]
        model = GridSearchCV(SVR(kernel='rbf'), tuned_parameters)
        model.fit(X_train, Y_train)
        print(model.best_estimator_)
        return model
