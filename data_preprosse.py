# -*- coding: utf-8 -*-
# @Time: 2023-11-03 10:00
# @Author: Haowei Chen
# @File: example.py
# @Software: PyCharm


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE


def scale_continuous_features(X_train, y_train, X_validationdata1, y_validationdata1,  X_testdata1, y_testdata1, X_testdata2, y_testdata2 ):
    # 获取数据类型为 float64 或 int64 的列名，这些通常是连续变量
    continuous_vars = X_train.columns.tolist()

    non_continuous_vars = [ 'age', 'Birth_history', 'abortion', 'IVF', 'PMH']
    # 移除已知的非连续变量
    continuous_vars = [var for var in continuous_vars if var not in non_continuous_vars]



    # 归一化数据
    X_train, X_validationdata1,X_testdata1, X_testdata2 = standardize_data(continuous_vars,'min-max',X_train, X_validationdata1, X_testdata1, X_testdata2)

    # 使用smote算法对训练集进行过采样
    sm = SMOTE(random_state=42,sampling_strategy=1)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print("查看返回数据类型：")
    print(type(X_train))
    print(type(y_train))



    return  X_train, y_train, X_validationdata1, y_validationdata1, X_testdata1, y_testdata1, X_testdata2, y_testdata2

def  standardize_data(continuous_vars,scale_mehtod,train_data, validation_data1,  test_data1, test_data2):


    # 对训练数据进行z-score标准化
    if scale_mehtod == 'z-score':
        scaler = StandardScaler()
    elif scale_mehtod == 'min-max':
        scaler = MinMaxScaler()

    train_data[continuous_vars] = scaler.fit_transform(train_data[continuous_vars])

    # 标准化其他数据集
    validation_data1[continuous_vars] = scaler.transform(validation_data1[continuous_vars])
    test_data1[continuous_vars] = scaler.transform(test_data1[continuous_vars])
    test_data2[continuous_vars] = scaler.transform(test_data2[continuous_vars])
    return train_data, validation_data1,test_data1, test_data2


