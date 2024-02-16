# -*- coding: utf-8 -*-
# @Time: 2023-11-03 10:00
# @Author: Haowei Chen
# @File: example.py
# @Software: PyCharm
import featuretools as ft
import pickle
import numpy as np


def calculate_feature_matrix(features, X_df, ID):
    # Create a new EntitySet
    es = ft.EntitySet(id='data')
    es = es.add_dataframe(dataframe_name='features', dataframe=X_df, index=ID,
                          make_index=True if ID not in X_df.columns else False)

    # Calculate feature matrix
    feature_matrix = ft.calculate_feature_matrix(features=features, entityset=es)

    # If the index of the feature matrix does not match the index of the original data frame, you may need to re-index
    feature_matrix = feature_matrix.reindex(X_df.index)

    feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Check if there is a NaN value and handle it
    feature_matrix.fillna(0, inplace=True)

    return feature_matrix


def feature_add_featuretools(X_train, y_train, X_validationdata1, y_validationdata1, X_testdata1, y_testdata1, X_testdata2, y_testdata2, ID):
    trans_primitives = [ 'multiply_numeric', 'divide_numeric']
    agg_primitives = ['sum', 'median', 'mean']
    # Create a new, empty EntitySet object
    es = ft.EntitySet(id='train_data')
    es = es.add_dataframe(dataframe_name='features', dataframe=X_train, index=ID,
                          make_index=True if ID not in X_train.columns else False)

    # Generate new features using DFS
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='features',
                                          max_depth=1,
                                          agg_primitives=agg_primitives,
                                          trans_primitives=trans_primitives)  # max_depth 是指生成特征的深度
    print(feature_defs)

    # Replace the original feature set with the generated features
    is_infinite = sum(np.isinf(feature_matrix).any())
    print(f"Does X_train contain infinite values? {is_infinite}")
    feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train = feature_matrix.reindex(index=X_train.index).fillna(0)


    # Save feature definition
    with open('./feature_definitions.pkl', 'wb') as f:
        pickle.dump(feature_defs, f)

    # Compute feature matrices for validation and test datasets
    X_validationdata1 = calculate_feature_matrix(feature_defs, X_validationdata1, ID)
    X_testdata1 = calculate_feature_matrix(feature_defs, X_testdata1, ID)
    X_testdata2 = calculate_feature_matrix(feature_defs, X_testdata2, ID)

    return X_train, y_train, X_validationdata1, y_validationdata1, X_testdata1, y_testdata1, X_testdata2, y_testdata2

