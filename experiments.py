import re

import os
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import ml_dataset_loader.datasets as data_loader
import pandas as pd
import time

test_size = 0.2
train_size = 0.8
random_seed = 7
#num_rounds = 5
num_rounds = 200
#global_num_rows = None
global_num_rows = 1000
early_stopping_rounds = 15

common_param = {'silent': 1, "reg_lambda": 0, "max_bin": 64, "debug_verbose":0}

df_execution_time = None

class Experiment:
    def __init__(self, name, objective, metric, load_data, param_set, num_rows=None):
        self.name = name
        self.objective = objective
        self.metric = metric
        self.load_data = load_data
        self.num_rows = num_rows
        self.param_set = param_set

    def run(self):
        # Create train/test/validation sets
        if global_num_rows is None:
            X, y = self.load_data(self.num_rows)
        else:
            X, y = self.load_data(global_num_rows)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=random_seed)
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)

        data = {"metric": self.metric, "results": {}}
        for opt_param in self.param_set:
            param = common_param.copy()
            param.update(opt_param)
            param["objective"] = self.objective
            param["eval_metric"] = self.metric

            if self.objective == "reg:linear":
                param["base_score"] = np.average(y_train)
            if self.objective == "multi:softmax":
                param["num_class"] = np.max(y_train) + 1
            if self.objective == "binary:logistic":
                pos = np.sum(y_train)
                param["scale_pos_weight"] = (len(y_train) - pos) / pos

            res = {}
            start = time.time()
            bst = xgb.train(param, dtrain, num_rounds,
                            evals=[(dtest, "test"), (dtrain, "train")],
                            early_stopping_rounds=early_stopping_rounds, evals_result=res)

            execution_time = round(time.time() - start, 2)
            df_execution_time.set_value(self.name, opt_param['name'], execution_time)
            # Extend results if early stopping
            test_metric = res['test'][self.metric]
            last = test_metric[-1]
            while len(test_metric) < num_rounds:
                test_metric.append(last)
            data["results"][param["name"]] = test_metric
            del bst

        # Save data
        if not os.path.exists("data"):
            os.makedirs("data")
        snake_name = re.sub(' ', '_', self.name).lower()
        f = open("data/" + snake_name + "_data.pkl", "wb")
        pickle.dump({self.name: data}, f)
        f.close()


experiment_param = [
    {'name': 'dct16/64', 'booster': 'gbdtl', 'discrete_transform': 'dct', 'max_coefficients': 16},
    {'name': 'haar16/64', 'booster': 'gbdtl', 'discrete_transform': 'haar', 'max_coefficients': 16},
    {'name': 'rp16/64', 'booster': 'gbdtl', 'discrete_transform': 'rp', 'max_coefficients': 16},
    {'name': 'identity64/64', 'booster': 'gbdtl', 'discrete_transform': 'identity', 'max_coefficients': 64},
    {'name': 'linear', 'booster': 'gblinear', 'updater': 'coord_descent'},
    {'name': 'stumps', 'booster': 'gbtree', 'tree_method': 'exact', 'max_depth': 1},
    {'name': 'gbm', 'booster': 'gbtree', 'tree_method': 'exact', 'max_depth': 6},
]

experiments = [
    Experiment("Synthetic classification", "binary:logistic", "error", data_loader.get_synthetic_classification, experiment_param, 10000),
    Experiment("Cover type", "multi:softmax", "merror", data_loader.get_cover_type, experiment_param),
    Experiment("YearPredictMSD", "reg:linear", "rmse", data_loader.get_year, experiment_param),
    Experiment("Higgs", "binary:logistic", "error", data_loader.get_higgs, experiment_param),
    Experiment("Adult", "binary:logistic", "error", data_loader.get_adult, experiment_param),
    Experiment("Wine Quality", "reg:linear", "rmse", data_loader.get_wine_quality, experiment_param),
]

df_execution_time = pd.DataFrame(columns=[p['name'] for p in experiment_param])

for exp in experiments:
    exp.run()

df_execution_time = df_execution_time.drop(columns=["rp16/64","identity64/64"])
print(df_execution_time.to_latex())

'''
transform_experiment_param = [
    {'name': 'dct16/64', 'booster': 'gbdtl', 'discrete_transform': 'dct', 'max_coefficients': 16},
    {'name': 'haar16/64', 'booster': 'gbdtl', 'discrete_transform': 'haar', 'max_coefficients': 16},
    {'name': 'identity16/16', 'booster': 'gbdtl', 'discrete_transform': 'identity', 'max_coefficients': 16,'max_bin':16},
    {'name': 'identity32/32', 'booster': 'gbdtl', 'discrete_transform': 'identity', 'max_coefficients': 32,'max_bin':32},
    {'name': 'identity48/48', 'booster': 'gbdtl', 'discrete_transform': 'identity', 'max_coefficients': 48,'max_bin':48},
    {'name': 'identity64/64', 'booster': 'gbdtl', 'discrete_transform': 'identity', 'max_coefficients': 64,'max_bin':64},
]

transform_experiments = [
    Experiment("Wine Quality # Transform comparison", "reg:linear", "rmse", data_loader.get_wine_quality, transform_experiment_param),
]

for exp in transform_experiments:
    exp.run()
'''

