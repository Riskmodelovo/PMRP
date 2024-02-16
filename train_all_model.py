# -*- coding: utf-8 -*-
# @Time: 2023-11-03 10:00
# @Author: Haowei Chen
# @File: example.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# train_all_models.py  This process uses train for training, validation1 for hyperparameter adjustment, and test1 and test2 for independent testing.
# # The clinical features are fixed, MAP IVF abortion BMI age is a required feature, and feature selection is performed on metabolites.

import argparse
import os
import data_read
from data_read import read_traindata_testdata
from data_preprosse import scale_continuous_features
from feature_add import feature_add_featuretools
from feature_select import select1_all_models
import feature_select
import models_space
import concurrent.futures
import os
import pandas as pd
import concurrent.futures



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test a model.')
    parser.add_argument('workPath', type=str, help='Path to work')
    parser.add_argument('traindata', type=str, help='Path to the training data file.')
    parser.add_argument('validationdata1', type=str, help='Path to the validation data file.')
    parser.add_argument('testdata1', type=str, help='Path to the testing data file.')
    parser.add_argument('testdata2', type=str, help='Path to the testing data file.')


    args = parser.parse_args()


    # Set new working directory
    new_working_directory = args.workPath
    os.chdir(new_working_directory)

    ID = ['hospital']
    group = 'group'
    pos_label = 'LPE'
    control = 'control'

    # 读取数据
    X_train, y_train, X_validationdata1, y_validationdata1, X_testdata1, y_testdata1, X_testdata2, y_testdata2 =  read_traindata_testdata(args.traindata, args.validationdata1, args.testdata1, args.testdata2, ID, group, pos_label, control)





    # feature add
    #X_train, y_train, X_validationdata1, y_validationdata1, X_testdata1, y_testdata1, X_testdata2, y_testdata2 = feature_add_featuretools(X_train, y_train, X_validationdata1, y_validationdata1,  X_testdata1, y_testdata1, X_testdata2, y_testdata2,'sample')
    feature_names = X_train.columns

    # scale feature
    X_train, y_train, X_validationdata1, y_validationdata1, X_testdata1, y_testdata1, X_testdata2, y_testdata2 = scale_continuous_features(X_train, y_train, X_validationdata1, y_validationdata1, X_testdata1, y_testdata1, X_testdata2, y_testdata2)

    sellect_feature = select1_all_models(X_train, y_train, X_validationdata1, y_validationdata1, X_testdata1, y_testdata1, X_testdata2,y_testdata2, feature_names)