# -*- coding: utf-8 -*-
# @Time: 2023-11-03 10:00
# @Author: Haowei Chen
# @File: example.py
# @Software: PyCharm

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from hyperopt import hp
class classficationModels:
    def __init__(self,model_name):
        self.model_name = model_name


     # Used to select a model based on the name of self.model_name
    def model_selection(self):
        if self.model_name == 'lasso':
            model = Lasso
            param_space = {
        'random_state': 42,
        'alpha': hp.uniform('alpha', 0.001, 0.1),
        'tol': hp.uniform('tol', 0.0001, 0.1),
    }
        elif self.model_name == 'lightgbm':
            model = LGBMClassifier
            param_space = {'n_jobs': 10,
                           'random_state': 42,
                           'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
                           'learning_rate': hp.uniform('learning_rate', 0.1, 1),
                           'lambda_l1': hp.uniform('lambda_l1', 5, 10),
                           'max_depth': hp.choice('max_depth', range(1, 5)),
                           }
        else:
            print('Please input the correct model name!')
        return model,param_space

