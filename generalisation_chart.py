import re

import os
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import ml_dataset_loader.datasets as data_loader
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler

test_size = 0.2
train_size = 0.8
random_seed = 7
#num_rounds = 5
num_rounds = 150
global_num_rows = None
#global_num_rows = 1000
early_stopping_rounds = 150

common_param = {'silent': 1, "reg_lambda": 0, "max_bin": 64, "debug_verbose":0}

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],  # use latex default serif font
})
plt.rc('axes', prop_cycle=(cycler('color', ['g', 'g', 'b', 'b', 'y', 'y','r','r']) +
                           cycler('linestyle', ['-', '--', '-', '--', '-', '--', '-', '--'])))

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
            bst = xgb.train(param, dtrain, num_rounds,
                            evals=[(dtest, "test"), (dtrain, "train")],
                            early_stopping_rounds=early_stopping_rounds, evals_result=res)

            # Extend results if early stopping
            plt.plot(res['train'][self.metric],
                     label=param['name'] + " - train")
            plt.plot(res['test'][self.metric],linestyle="--",
                 label=param['name'] + " - test")
            del bst


        plt.xlabel('iterations')
        plt.ylabel(data["metric"])
        plt.legend()
        plt.title(self.name)
        plt.savefig('figures/generalisation.pgf', bbox_inches='tight')
        plt.savefig('figures/generalisation.png', bbox_inches='tight')


experiment_param = [
    {'name': 'dct8/64', 'booster': 'gbdtl', 'discrete_transform': 'dct', 'max_coefficients': 8},
    {'name': 'dct16/64', 'booster': 'gbdtl', 'discrete_transform': 'dct', 'max_coefficients': 16},
    {'name': 'dct32/64', 'booster': 'gbdtl', 'discrete_transform': 'dct', 'max_coefficients': 32},
    {'name': 'dct48/64', 'booster': 'gbdtl', 'discrete_transform': 'dct', 'max_coefficients': 48},
]

experiments = [
    Experiment("Synthetic classification", "binary:logistic", "error", data_loader.get_synthetic_classification, experiment_param, 10000),
    #Experiment("Wine Quality", "reg:linear", "rmse", data_loader.get_wine_quality, experiment_param),
]

for exp in experiments:
    exp.run()
