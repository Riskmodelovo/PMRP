# -*- coding: utf-8 -*-
# @Time: 2023-11-03 10:00
# @Author: Haowei Chen
# @File: example.py
# @Software: PyCharm

import pandas as pd
import numpy as np

def preprocess_data(data_path, ID, group, pos_label, control):
    data = pd.read_csv(data_path,index_col=0)
    drop_clin = 0
    if drop_clin == 1:
        # Remove all clinical featuresBMI,Birth_history,GA_sampling,IVF,MAP,PMH,abortion,age
        data.drop(['BMI','Birth_history','GA_sampling','IVF','MAP','PMH','abortion','age'], axis=1, inplace=True)
    elif drop_clin == 2:
         # Remove Birth_history, GA_sampling, and PMH. These features do not participate in the final model training.
        data.drop(['Birth_history','GA_sampling','PMH'], axis=1, inplace=True)



    # Filter samples based on group column
    data = data[data[group].isin([pos_label, control])]



    # Remove ID column
    data.drop(ID, axis=1, inplace=True)

    # Assume that there is a column named 'group' in the data as the target variable
    X = data.drop(group, axis=1)
    y = data[group]

    # Replace LPE in Y with 1 and control with 0, so pos_bale is the disease
    y = y.replace([pos_label, control], [1, 0])

    # Convert y to a one-dimensional array
    y = y.values.ravel()


    return X, y






def read_traindata_testdata(traindata_path, validationdata1_path,  testdata1_path, testdata2_path,
                            ID, group, pos_label, control):
    X_train, y_train = preprocess_data(traindata_path, ID, group, pos_label, control)
    X_validationdata1, y_validationdata1 = preprocess_data(validationdata1_path, ID, group, pos_label, control)
    X_testdata1, y_testdata1 = preprocess_data(testdata1_path, ID, group, pos_label, control)
    X_testdata2, y_testdata2 = preprocess_data(testdata2_path, ID, group, pos_label, control)

    print(f'Reading data is completed, the data is{len(X_train.columns)}')


    return X_train, y_train, X_validationdata1, y_validationdata1, X_testdata1, y_testdata1, X_testdata2, y_testdata2


