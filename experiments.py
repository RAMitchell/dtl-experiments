import re

import os
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split

import ml_dataset_loader.datasets as data_loader

test_size = 0.2
train_size = 0.8
random_seed = 7
#num_rounds = 1000
num_rounds = 150
early_stopping_rounds = 10

common_param = {'silent': 1, "reg_lambda": 0, "max_bin": 64}
experiment_param = [
    {'name': 'dct16/64', 'booster': 'gbdct', 'regularising_transform': 'dct', 'max_coefficients': 16},
    {'name': 'dct32/64', 'booster': 'gbdct', 'regularising_transform': 'dct', 'max_coefficients': 32},
    {'name': 'dct64/64', 'booster': 'gbdct', 'regularising_transform': 'dct', 'max_coefficients': 32},
    {'name': 'dct32/128', 'booster': 'gbdct', 'regularising_transform': 'dct', 'max_coefficients': 32,'max_bin': 128},
    {'name': 'dct64/128', 'booster': 'gbdct', 'regularising_transform': 'dct', 'max_coefficients': 64,'max_bin': 128},
    {'name': 'dct128/128', 'booster': 'gbdct', 'regularising_transform': 'dct', 'max_coefficients': 128,'max_bin': 128},
    #{'name': 'haar32/64', 'booster': 'gbdct', 'regularising_transform': 'haar', 'max_coefficients': 32},
    #{'name': 'dct8/64', 'booster': 'gbdct', 'regularising_transform': 'dct', 'max_coefficients': 8},
    #{'name': 'haar8/64', 'booster': 'gbdct', 'regularising_transform': 'haar', 'max_coefficients': 8},
    #{'name': 'rp16/64', 'booster': 'gbdct', 'regularising_transform': 'rp', 'max_coefficients': 16},
    #{'name': 'linear', 'booster': 'gblinear', 'updater': 'coord_descent'},
    #{'name': 'stumps', 'booster': 'gbtree', 'tree_method': 'gpu_hist', 'max_depth': 1},
    #{'name': 'gbm', 'booster': 'gbtree', 'tree_method': 'gpu_hist', 'max_depth': 6},
]


class Experiment:
    def __init__(self, name, objective, metric, load_data):
        self.name = name
        self.objective = objective
        self.metric = metric
        self.load_data = load_data

    def run(self, num_rows=None):
        # Create train/test/validation sets
        # Don't use full airline dataset
        if self.name == "Airline" and num_rows is None:
            num_rows = 10000000
        X, y = self.load_data(num_rows)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=random_seed)
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)

        data = {"metric": self.metric, "results": {}}
        for opt_param in experiment_param:
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
            bst = xgb.train(param, dtrain, num_rounds,
                            evals=[(dtest, "test"),(dtrain, "train")],
                            early_stopping_rounds=early_stopping_rounds, evals_result=res)
            data["results"][param["name"]] = res['test'][self.metric]
            del bst

        # Save data
        if not os.path.exists("data"):
            os.makedirs("data")
        snake_name = re.sub(' ', '_', self.name).lower()
        f = open("data/" + snake_name + "_data.pkl", "wb")
        pickle.dump({self.name: data}, f)
        f.close()


experiments = [
    Experiment("Synthetic classification", "binary:logistic", "error", data_loader.get_synthetic_classification),
    Experiment("Cover type", "multi:softmax", "merror", data_loader.get_cover_type),
    Experiment("YearPredictMSD", "reg:linear", "rmse", data_loader.get_year),
    Experiment("Higgs", "binary:logistic", "error", data_loader.get_higgs),
    Experiment("Airline", "binary:logistic", "error", data_loader.get_airline),
]

#num_rows = None
num_rows = 100000
for exp in experiments:
    exp.run(num_rows)
