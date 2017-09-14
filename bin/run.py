import argparse
import os
import subprocess
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from scipy.stats.stats import pearsonr

#import util
import config


# Constants
MAIN = config.MAIN_FOLDER


# Parse input
parser = argparse.ArgumentParser(description='Train and test models')
parser.add_argument('-m', '--model', type=str, default='ridge')
parser.add_argument('-f', '--folds', nargs='*', type=int, default=[0])
args = parser.parse_args()


