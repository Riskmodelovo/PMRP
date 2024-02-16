# -*- coding: utf-8 -*-
# @Time: 2023-11-03 10:00
# @Author: Haowei Chen
# @File: example.py
# @Software: PyCharm
import statistics

import numpy.random
from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, recall_score

import models_space
import os
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from hyperopt import tpe, Trials, fmin, STATUS_OK, space_eval, plotting, atpe,rand
from models_space import classficationModels
import numpy as np
import warnings
numpy.warnings = warnings
from early_stop import no_progress_loss
import lightgbm as lgb
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from scipy.stats import norm

from sklearn.metrics import classification_report
import random

random.seed(42)
np.random.seed(42)



class ModelOptimizer:
    def __init__(self, model_name,folder, X_train, y_train, X_val1, y_val1, X_test1, y_test1, X_test2, y_test2):
        self.model_name = model_name
        classmodel  = classficationModels(self.model_name)
        self.model_class, self.space = classmodel.model_selection()
        self.folder = folder
        self.X_train = X_train
        self.y_train = y_train
        self.X_val1 = X_val1
        self.y_val1 = y_val1
        self.X_test1 = X_test1
        self.y_test1 = y_test1
        self.X_test2 = X_test2
        self.y_test2 = y_test2
        self.workpath = os.getcwd()
        self.save_path = os.path.join(self.workpath,folder)
        self.save_path1= os.path.join(self.save_path, 'model_result')
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.save_path1, exist_ok=True)




    # Use validation1 for hyperparameter tuning, test1, test2 for testing
    def objective1(self, params):
        model = self.model_class(**params)
        model.fit(self.X_train, self.y_train)
        val1_score = self.calu_auc(model, self.X_val1, self.y_val1)

        return {'loss': -val1_score,'status': STATUS_OK}

    def optimize1(self):

        self.save_X()
        trials = Trials()
        early_stop_fn = no_progress_loss(iteration_stop_count= 300, percent_increase=0.01)
        best_params = fmin(fn=self.objective1, space=self.space, algo= tpe.suggest, max_evals=1000, trials=trials,rstate=np.random.default_rng(seed=42),early_stop_fn=early_stop_fn)
        best_params = space_eval(self.space, best_params)

        # Save the best parameters, parameter adjustment process and optimization diagram
        self.save_with_timestamp(best_params, 'best_params')
        self.save_with_timestamp(trials, 'trials')
        self.save_optimization_plot(trials)



        # Build models with optimal parameters and calculate AUC on each dataset
        print("Start evaluating the model")
        auc_df = self.evaluate_model1(best_params)

        # Save confusion matrix
        threshold_class = self.confux_matrix(best_params)

        # Calculate 95 CI confidence intervals on each data set
        mean = self.bootstrap_auc(best_params, self.X_train, self.y_train, 'train')
        mean_val1 = self.bootstrap_auc(best_params, self.X_val1, self.y_val1, 'val1')
        mean = self.bootstrap_auc(best_params, self.X_test1, self.y_test1, 'test1')
        mean = self.bootstrap_auc(best_params, self.X_test2, self.y_test2, 'test2')


        return best_params,auc_df,0






    def calu_auc(self,model,X,y):
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1]
        elif hasattr(model,'decision_function'):
            probabilities= model.decision_function(X)
        else:
           print('Model does not support predicted probabilities')
           probabilities = model.predict(X)
        auc = roc_auc_score(y, probabilities)
        return auc



    def save_with_timestamp(self, obj, filename):
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename_with_time = f"{self.model_name}_{filename}.pkl"
        full_path = os.path.join(self.save_path, filename_with_time)
        with open(full_path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Saved {filename} to {full_path}")

    def save_optimization_plot(self, trials):
        plotting.main_plot_history(trials, do_show=False)
        plt.savefig(os.path.join(self.save_path, f'{self.model_name}_optimization_history.png'))
        plt.close()


    def evaluate_model1(self, best_params):
        model = self.model_class(**best_params)
        model.fit(self.X_train, self.y_train)

        train_auc = self.calu_auc(model, self.X_train, self.y_train)
        valid1_auc = self.calu_auc(model, self.X_val1, self.y_val1)
        test1_auc = self.calu_auc(model, self.X_test1, self.y_test1)
        test2_auc = self.calu_auc(model, self.X_test2, self.y_test2)

        # Calculate the roc curve of test1 and test2
        fpr1, tpr1, thresholds = roc_curve(self.y_test1, model.predict_proba(self.X_test1)[:, 1])
        # Save fpr1 and tpr1, thresholds to files
        fpr1_tpr1_thresholds = pd.DataFrame({'fpr1': fpr1, 'tpr1': tpr1, 'thresholds': thresholds})
        fpr1_tpr1_thresholds.to_csv(os.path.join(self.save_path1, f'{self.model_name}_fpr1_tpr1_thresholds.csv'), index=False)

        fpr2, tpr2, thresholds = roc_curve(self.y_test2, model.predict_proba(self.X_test2)[:, 1])

        target_fpr = 0.1  # 1 - 0.9
        closest_index = np.argmin(np.abs(fpr1 - target_fpr))
        closest_fpr1, closest_tpr1 = fpr1[closest_index], tpr1[closest_index]
        closest_index = np.argmin(np.abs(fpr2 - target_fpr))
        closest_fpr2, closest_tpr2 = fpr2[closest_index], tpr2[closest_index]


        auc_data = {
            'Train AUC': train_auc,
            'Valid AUC 1': valid1_auc,
            'test1 AUC 2': test1_auc,
            'Screen AUC 1': test2_auc,
            'Screen  TPR1': closest_tpr1,
        }


        # Convert to DataFrame
        auc_df = pd.DataFrame([auc_data])

        # Save to CSV
        auc_filename = os.path.join(self.save_path, f'{self.model_name}_auc_scores.csv')
        auc_df.to_csv(auc_filename, index=False)
        print(f'AUC scores saved to {auc_filename}')

        # save model
        self.save_model(model)

        # The predicted results and y_train real results saved on self.X_train
        self.save_predict_result(model)

        self.threshold_save_feature(model)

        return auc_df







    def save_model(self, model):
        model_filename = os.path.join(self.save_path1, f'{self.model_name}.joblib')
        dump(model, model_filename)
        print(f'Model saved to {model_filename}')

    def save_predict_result(self, model):
        # The predicted results and y_train real results saved on self.X_train
        train_pred = model.predict(self.X_train)
        train_pred_proba = 0
        if hasattr(model, 'predict_proba'):
            train_pred_proba = model.predict_proba(self.X_train)[:, 1]
        elif hasattr(model, 'decision_function'):
            train_pred_proba = model.decision_function(self.X_train)
        train_pred_df = pd.DataFrame({'pred': train_pred, 'true': self.y_train, 'pred_proba': train_pred_proba})
        train_pred_df.to_csv(os.path.join(self.save_path1, f'{self.model_name}_train_pred.csv'), index=False)

        # The predicted results and y_val1 real results saved on self.X_val1
        val1_pred = model.predict(self.X_val1)
        val1_pred_proba = 0
        if hasattr(model, 'predict_proba'):
            val1_pred_proba = model.predict_proba(self.X_val1)[:, 1]
        elif hasattr(model, 'decision_function'):
            val1_pred_proba = model.decision_function(self.X_val1)
        val1_pred_df = pd.DataFrame({'pred': val1_pred, 'true': self.y_val1, 'pred_proba': val1_pred_proba})
        val1_pred_df.to_csv(os.path.join(self.save_path1, f'{self.model_name}_val1_pred.csv'), index=False)



        # The predicted results and y_test1 real results saved on self.X_test1
        test1_pred = model.predict(self.X_test1)
        test1_pred_proba = 0
        if hasattr(model, 'predict_proba'):
            test1_pred_proba = model.predict_proba(self.X_test1)[:, 1]
        elif hasattr(model, 'decision_function'):
            test1_pred_proba = model.decision_function(self.X_test1)
        test1_pred_df = pd.DataFrame({'pred': test1_pred, 'true': self.y_test1, 'pred_proba': test1_pred_proba})
        test1_pred_df.to_csv(os.path.join(self.save_path1, f'{self.model_name}_test1_pred.csv'), index=False)

        # The predicted results and y_test2 real results saved on self.X_test2
        test2_pred = model.predict(self.X_test2)
        test2_pred_proba = 0
        if hasattr(model, 'predict_proba'):
            test2_pred_proba = model.predict_proba(self.X_test2)[:, 1]
        elif hasattr(model, 'decision_function'):
            test2_pred_proba = model.decision_function(self.X_test2)
        test2_pred_df = pd.DataFrame({'pred': test2_pred, 'true': self.y_test2, 'pred_proba': test2_pred_proba})
        test2_pred_df.to_csv(os.path.join(self.save_path1, f'{self.model_name}_test2_pred.csv'), index=False)









    def threshold_save_feature(self, rf):
        feature_names = self.X_train.columns
        rf_importance = 0
        try:
            rf_importance = rf.feature_importances_
        except AttributeError:
            try:
                rf_importance = rf.coef_.tolist()[0]
            except AttributeError:
                rf_importance = permutation_importance(rf, self.X_train, self.y_train, n_repeats=10, random_state=42, n_jobs=-1)
                rf_importance = rf_importance.importances_mean

        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_importance}).sort_values(by='importance', ascending=False)

        save_path = os.path.join(self.save_path1,f'{self.model_name}_importance.csv')
        rf_importance.to_csv(save_path, index=False)

        return rf_importance

    def confux_matrix(self,best_params):
        # Save the confusion matrix with the default 0.5 as the threshold
        model = self.model_class(**best_params)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_train)
        cm = confusion_matrix(self.y_train, y_pred)
        result = classification_report(self.y_train, y_pred,output_dict=True)
        result = pd.DataFrame(result).T
        cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
        cm_df = pd.concat([cm_df, result], axis=1)
        save_path = os.path.join(self.save_path1, f'{self.model_name}_train_confusion_matrix.csv')
        cm_df.to_csv(save_path)

        y_pred = model.predict(self.X_val1)
        cm = confusion_matrix(self.y_val1, y_pred)
        result = classification_report(self.y_val1, y_pred,output_dict=True)
        cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
        result = pd.DataFrame(result).T
        cm_df = pd.concat([cm_df, result], axis=1)
        save_path = os.path.join(self.save_path1, f'{self.model_name}_val1_confusion_matrix.csv')
        cm_df.to_csv(save_path)


        y_pred = model.predict(self.X_test1)
        cm = confusion_matrix(self.y_test1, y_pred)
        cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
        result = classification_report(self.y_test1, y_pred,output_dict=True)
        result = pd.DataFrame(result).T
        cm_df = pd.concat([cm_df, result], axis=1)
        save_path = os.path.join(self.save_path1, f'{self.model_name}_test1_confusion_matrix.csv')
        cm_df.to_csv(save_path)

        y_pred = model.predict(self.X_test2)
        cm = confusion_matrix(self.y_test2, y_pred)
        cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
        result = classification_report(self.y_test2, y_pred,output_dict=True)
        result = pd.DataFrame(result).T
        cm_df = pd.concat([cm_df, result], axis=1)
        save_path = os.path.join(self.save_path1, f'{self.model_name}_test2_confusion_matrix.csv')
        cm_df.to_csv(save_path)

      # 保存混淆矩阵以test2最佳阈值为阈值
        y_pro = model.predict_proba(self.X_test2)[:, 1]

        # 找到x_test2的最佳阈值
        fpr_lr, tpr_lr, threld = roc_curve(self.y_test2,y_pro)

        Youden_index = np.argmax(tpr_lr - fpr_lr)  # Only the first occurrence is returned.
        optimal_threshold = threld[Youden_index]
        point = [fpr_lr[Youden_index], tpr_lr[Youden_index]]
        print('test2最佳阈值为：')
        print(point)

        # 使用最佳分类点重新预测数据，并保存confusion_matix
        y_pred = (y_pro > optimal_threshold).astype(int)
        cm = confusion_matrix(self.y_test2, y_pred)
        cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
        result = classification_report(self.y_test2, y_pred,output_dict=True)
        result = pd.DataFrame(result).T
        cm_df = pd.concat([cm_df, result], axis=1)
        save_path = os.path.join(self.save_path1, f'{self.model_name}_test2_confusion_matrix_optimal.csv')
        cm_df.to_csv(save_path)

        y_pro = model.predict_proba(self.X_test1)[:, 1]
        y_pred = (y_pro > optimal_threshold).astype(int)
        cm = confusion_matrix(self.y_test1, y_pred)
        cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
        result = classification_report(self.y_test1, y_pred,output_dict=True)
        result = pd.DataFrame(result).T
        cm_df = pd.concat([cm_df, result], axis=1)
        save_path = os.path.join(self.save_path1, f'{self.model_name}_test1_confusion_matrix_optimal.csv')
        cm_df.to_csv(save_path)



        y_pro = model.predict_proba(self.X_val1)[:, 1]
        y_pred = (y_pro > optimal_threshold).astype(int)
        cm = confusion_matrix(self.y_val1, y_pred)
        cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
        result = classification_report(self.y_val1, y_pred,output_dict=True)
        result = pd.DataFrame(result).T
        cm_df = pd.concat([cm_df, result], axis=1)
        save_path = os.path.join(self.save_path1, f'{self.model_name}_val1_confusion_matrix_optimal.csv')
        cm_df.to_csv(save_path)

        y_pro = model.predict_proba(self.X_train)[:, 1]
        y_pred = (y_pro > optimal_threshold).astype(int)
        cm = confusion_matrix(self.y_train, y_pred)
        cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
        result = classification_report(self.y_train, y_pred,output_dict=True)
        result = pd.DataFrame(result).T
        cm_df = pd.concat([cm_df, result], axis=1)
        save_path = os.path.join(self.save_path1, f'{self.model_name}_train_confusion_matrix_optimal.csv')
        cm_df.to_csv(save_path)


        return optimal_threshold


    # Use Bootstrap method to calculate AUC confidence interval
    def bootstrap_auc(self,best_params,X,y,data_name):
        model = self.model_class(**best_params)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred)
        print(f"Raw AUC: {auc}")

        # 使用自助法计算 AUC 置信区间
        n_iterations = 10000
        auc_scores = []
        sensitivity_scores = []
        specificity_scores = []
        fpr = []
        tpr = []
        ppv = []
        npv = []

        # 计算y中1/(0+1)的
        prevalence = np.sum(y) / len(y)



        for i in range(n_iterations):
            # 自助采样
            Xb, yb = resample(X, y,stratify=y)
            yb_pred = model.predict_proba(Xb)[:, 1]
            auc_score = roc_auc_score(yb, yb_pred)

            # 计算test1和test2的roc曲线
            fpr1, tpr1, thresholds = roc_curve(yb, yb_pred)
            target_fpr = 0.1  # 1 - 0.9
            closest_index = np.argmin(np.abs(fpr1 - target_fpr))
            closest_fpr1, closest_tpr1 = fpr1[closest_index], tpr1[closest_index]
            fpr.append(closest_fpr1)
            tpr.append(closest_tpr1)


            # 计算recall
            sensitivity_score = recall_score(yb, yb_pred>0.5)
            sensitivity_scores.append(sensitivity_score)

            # 计算specificity
            specificity_score = recall_score(yb, yb_pred>0.5,pos_label=0)
            specificity_scores.append(specificity_score)

            auc_scores.append(auc_score)

            ppv_score = (sensitivity_score * prevalence) / ((sensitivity_score * prevalence) + ((1 - specificity_score) * (1 - prevalence)))
            ppv.append(ppv_score)

            npv_score = (specificity_score * (1 - prevalence)) / ((specificity_score * (1 - prevalence)) + ((1 - sensitivity_score) * prevalence))
            npv.append(npv_score)

        # 计算 95% 置信区间
        mean,std_dev = norm.fit(auc_scores)
        confidence_level = 0.95
        lower ,upper = norm.interval(confidence_level,loc=mean,scale=std_dev)
        print(f"auc 均值：{mean}, 95% 置信区间: {lower} 到 {upper}")

        # 计算sensitivity 95% 置信区间
        mean_sensitive,std_dev = norm.fit(sensitivity_scores)
        lower_sensitive ,upper_sensitive = norm.interval(confidence_level,loc=mean_sensitive,scale=std_dev)
        print(f"sensitivity 均值：{mean_sensitive}, 95% 置信区间: {lower_sensitive} 到 {upper_sensitive}")

        # 计算specificity 95% 置信区间
        mean_specificity,std_dev = norm.fit(specificity_scores)
        lower_specificity ,upper_specificity = norm.interval(confidence_level,loc=mean_specificity,scale=std_dev)
        print(f"specificity 均值：{mean_specificity}, 95% 置信区间: {lower_specificity} 到 {upper_specificity}")

        # 计算tpr 95% 置信区间sentivity
        mean_tpr,std_dev = norm.fit(tpr)
        lower_tpr ,upper_tpr = norm.interval(confidence_level,loc=mean_tpr,scale=std_dev)
        print(f"tpr 均值：{mean_tpr}, 95% 置信区间: {lower_tpr} 到 {upper_tpr}")

        # 计算fpr 95% 置信区间
        mean_fpr,std_dev = norm.fit(fpr)
        lower_fpr ,upper_fpr = norm.interval(confidence_level,loc=mean_fpr,scale=std_dev)
        print(f"fpr 均值：{mean_fpr}, 95% 置信区间: {lower_fpr} 到 {upper_fpr}")

        # 计算ppv 95% 置信区间
        mean_ppv,std_dev = norm.fit(ppv)
        lower_ppv ,upper_ppv = norm.interval(confidence_level,loc=mean_ppv,scale=std_dev)
        print(f"ppv 均值：{mean_ppv}, 95% 置信区间: {lower_ppv} 到 {upper_ppv}")

        # 计算npv 95% 置信区间
        mean_npv,std_dev = norm.fit(npv)
        lower_npv ,upper_npv = norm.interval(confidence_level,loc=mean_npv,scale=std_dev)
        print(f"npv 均值：{mean_npv}, 95% 置信区间: {lower_npv} 到 {upper_npv}")


        # 将 auc_scores, sensitivity_scores, specificity_scores 保存到csv

        auc_scores_df = pd.DataFrame({
            'AUC Scores': auc_scores,
            'Sensitivity Scores': sensitivity_scores,
            'Specificity Scores': specificity_scores,
            'fpr':fpr,
            'tpr':tpr,
            'ppv':ppv,
            'npv':npv
        })
        save_path = os.path.join(self.save_path1, f'{self.model_name}_{data_name}_auc_sen_spe_scores.csv')
        auc_scores_df.to_csv(save_path, index=False)

        CI = {
            'auc_cl':[mean,lower,upper],
            'sensitivity_cl':[mean_sensitive,lower_sensitive,upper_sensitive],
            'specificity_cl':[mean_specificity,lower_specificity,upper_specificity],
            'tpr_cl':[mean_tpr,lower_tpr,upper_tpr],
            'fpr_cl':[mean_fpr,lower_fpr,upper_fpr],
            'ppv_cl':[mean_ppv,lower_ppv,upper_ppv],
            'npv_cl':[mean_npv,lower_npv,upper_npv]
        }
        save_path = os.path.join(self.save_path1, f'{self.model_name}_{data_name}_auc_sen_spe_95CI.csv')
        # 将Cl保存
        CI = pd.DataFrame([CI])

        CI.to_csv(save_path,index=False)

        return mean

    def save_X(self):
        # 保存X_train,X_val1,X_val2,X_test1,X_test2
        self.X_train.to_csv(os.path.join(self.save_path, f'{self.model_name}_X_train.csv'), index=False)
        self.X_val1.to_csv(os.path.join(self.save_path, f'{self.model_name}_X_val1.csv'), index=False)
        self.X_test1.to_csv(os.path.join(self.save_path, f'{self.model_name}_X_test1.csv'), index=False)
        self.X_test2.to_csv(os.path.join(self.save_path, f'{self.model_name}_X_test2.csv'), index=False)























