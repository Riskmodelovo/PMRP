# -*- coding: utf-8 -*-
# @Time: 2023-11-03 10:00
# @Author: Haowei Chen
# @File: example.py
# @Software: PyCharm


from boruta import BorutaPy
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, RFECV, RFE
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import dump, load
import numpy as np


from hyperopt_models import ModelOptimizer
from sklearn.linear_model import Lasso, LogisticRegression, LassoCV
from models_space import classficationModels
import matplotlib.pyplot as plt
import os
import logging

# Show all columns
pd.set_option('display.max_columns', None)

# show all rows
pd.set_option('display.max_rows', None)


# Used to perform hyperparameter tuning again after screening metabolites and adding clinical information.
add_clinc = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger1 = logging.getLogger('task1')
logger1.addHandler(logging.FileHandler('task1.log'))




def optimize_model1(model_name,folder,X_train, y_train, X_validationdata1, y_validationdata1,  X_testdata1, y_testdata1, X_testdata2, y_testdata2, select_feature):
    logger1.info(f'Start {model_name} for hyperparameter tuning')
    optimizer = ModelOptimizer(model_name, folder, X_train[select_feature], y_train, X_validationdata1[select_feature], y_validationdata1, X_testdata1[select_feature], y_testdata1, X_testdata2[select_feature], y_testdata2)
    params,auc_df,mean_auc = optimizer.optimize1()
    logger1.info(f"The auc on the validation set is {mean_auc}")
    logger1.info(f'{model_name} hyperparameter tuning completed')
    logger1.info(f'best params:{params}')
    logger1.info(f'{auc_df}')

    return params





def select1_all_models(X_train, y_train, X_validationdata1, y_validationdata1, X_testdata1, y_testdata1, X_testdata2, y_testdata2, feature_names):
    logger1.info("Select1 is starting...")

    file_path = './feature_select1_hyper/common_features.txt'

    if os.path.exists(file_path) and add_clinc:
        select_feature = pd.read_table(file_path, header=None)[0].tolist()
        select_feature +=  ['MAP','IVF','abortion','age','BMI']
        select_feature = list(set(select_feature))
    elif os.path.exists(file_path):
        select_feature = pd.read_table(file_path, header=None)[0].tolist()

    else:
        select_feature = feature_select1(X_train, y_train,feature_names)

    model_names = ['lightgbm']

    logger1.info(f'{select_feature}')
    logger1.info('Feature selection completed select1')

    folder = 'selected1_all_models_hyper'
    # Using process pools to perform model optimization
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(optimize_model1,model_name,folder, X_train, y_train, X_validationdata1, y_validationdata1, X_testdata1, y_testdata1, X_testdata2, y_testdata2, select_feature) for model_name in model_names]
        for future in as_completed(futures):
            future.result()

    logger1.info("Select1 is completed.")


    return select_feature







def threshold_save_feature(rf,feature_names,save_path):
    try:
        model = SelectFromModel(rf, prefit=True,threshold=0,max_features=10)
        rf_selected = feature_names[model.get_support()]
        rf_importance = rf.feature_importances_[model.get_support()]
    except AttributeError:
        model = SelectFromModel(rf, prefit=True,threshold=0,max_features=10)
        rf_selected = feature_names[model.get_support()]
        rf_importance = rf.coef_[model.get_support()]
    rf_importance = pd.DataFrame({
        'feature': rf_selected,
        'importance': rf_importance}).sort_values(by='importance', ascending=False)
    rf_importance.to_csv(save_path, index=False)
    return rf_importance


def feature_select1(X_train, y_train,feature_names):
    folder = 'feature_select1_hyper'
    os.makedirs(f'./{folder}', exist_ok=True)
    # Adaptive LASSO positive=True
    lasso = LassoCV(alphas=[0.1, 1], random_state=42, positive=True)
    lasso.fit(X_train, y_train)
    auc = calu_auc(lasso,X_train,y_train)
    logger1.info(f'lasso traindata auc:{auc}')
    lasso_importance = threshold_save_feature(lasso,feature_names,f'./{folder}/lasso_selected_features_importance.csv')

    # Univariate analysis AUC value
    feature_auc = {}
    for feature in feature_names:
        auc_score = roc_auc_score(y_train, X_train[feature])
        feature_auc[feature] = auc_score
    feature_auc = pd.DataFrame.from_dict(feature_auc, orient='index',columns=['importance']).sort_values(by='importance', ascending=False)
    # 只保留feature_auc中importance大于mean值的特征
    feature_auc = feature_auc[feature_auc['importance'] > feature_auc['importance'].mean()]
    feature_auc.index.name = 'feature'
    feature_auc.to_csv(f'./{folder}/single_AUC_feature_importance.csv')



    # GBDT
    gbdt = GradientBoostingClassifier(random_state=42, max_depth=5)
    gbdt.fit(X_train, y_train)
    auc = calu_auc(gbdt,X_train,y_train)
    logger1.info(f'gbdt traindata auc:{auc}')
    gbdt_importance = threshold_save_feature(gbdt,feature_names,f'./{folder}/gbdt_selected_features_importance.csv')

    #random forest
    rf = RandomForestClassifier(random_state=42,  class_weight='balanced', max_depth=5)
    rf.fit(X_train, y_train)
    auc = calu_auc(rf,X_train,y_train)
    logger1.info(f'rf traindata auc:{auc}')
    rf_importance = threshold_save_feature(rf,feature_names,f'./{folder}/rf_selected_features_importance.csv')


    lasso_features = set(lasso_importance['feature'])
    auc_features = set(feature_auc.index)
    gbdt_features = set(gbdt_importance['feature'])
    rf_features = set(rf_importance['feature'])

    # 使用集合的交集操作获取特征列的交集
    common_features = lasso_features & auc_features & gbdt_features & rf_features

    # 转换成列表
    common_features_list = list(common_features)

    with open(f'./{folder}/common_features.txt', 'w') as f:
        f.write('\n'.join(common_features_list))

    return common_features_list




def calu_auc(model,X,y):
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1]
        elif hasattr(model,'decision_function'):
            probabilities= model.decision_function(X)
        else:
            print('模型不支持预测概率')
            probabilities = model.predict(X)
        auc = roc_auc_score(y, probabilities)
        return auc




