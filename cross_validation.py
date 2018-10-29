import re
import os
from dask.distributed import Client

client = Client('localhost:8786')
import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
# from sklearn.model_selection import GridSearchCV
from dask_ml.model_selection import GridSearchCV
import ml_dataset_loader.datasets as data_loader
import pandas as pd

test_size = 0.2
train_size = 0.8
random_seed = 7
#num_rounds = 5
num_rounds = 200
#global_num_rows = None
global_num_rows = 1000000
early_stopping_rounds = 15
cv_folds = 5
logspace_steps = 100
l1_range = np.logspace(-8,1,logspace_steps)

bin_step_size = 2

common_param = {'silent': 1, "reg_lambda": 0, "max_bin": 64, "debug_verbose": 0, 'learning_rate': 0.5,
                'n_estimators': num_rounds, 'tree_method': 'exact'}

df_cv_results = None


class Experiment:
    def __init__(self, name, objective, metric, load_data, param_set, num_rows=None):
        self.name = name
        self.objective = objective
        self.metric = metric
        self.load_data = load_data
        self.num_rows = num_rows
        self.param_set = param_set
        # Create train/test/validation sets
        if global_num_rows is None:
            X, y = self.load_data(self.num_rows)
        else:
            X, y = self.load_data(global_num_rows)

        # Standardise
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                      test_size=test_size,
                                                                      random_state=random_seed)
        X_scaler = StandardScaler()
        self.X_train = X_scaler.fit_transform(X_train)
        self.X_test = X_scaler.transform(X_test)

    def run(self):
        print("Running dataset: " + self.name)
        for exp_param in self.param_set:
            param = common_param.copy()
            param.update(exp_param["xgb_params"])
            cv_param = exp_param["search parameter"]
            print("Running cv grid for: " + str(param))
            grid_param = cv_param

            # param["reg_alpha"] = 0.001
            # model = xgb.XGBRegressor(**param)
            # model.fit(X,y)
            # print(model.get_booster().get_dump()[0])
            # exit(0)
            model = xgb.XGBRegressor(**param) if self.objective == "reg:linear" else xgb.XGBClassifier(**param)
            clf = GridSearchCV(model, grid_param, cv=5, n_jobs=-1)
            # with parallel_backend('dask'):
            clf.fit(self.X_train, self.y_train)
            # clf.fit(X_train, y_train)
            param.update(clf.best_params_)
            model = xgb.XGBRegressor(**param) if self.objective == "reg:linear" else xgb.XGBClassifier(**param)
            model.fit(self.X_train, self.y_train)
            pred = model.predict(self.X_test)
            if isinstance(model, xgb.XGBRegressor):
                score = np.sqrt(metrics.mean_squared_error(self.y_test, pred))
            else:
                score = 1.0 - metrics.accuracy_score(self.y_test, pred)
            df_cv_results.at[exp_param['name'], (self.name, self.metric)] = "{0:.4g}".format(score)
            best_param_value = list(clf.best_params_.values())[0]
            best_param_string = "{0:.4g}".format(
                best_param_value) if best_param_value > 0.1 or best_param_value == 0.0 else "{0:.4e}".format(
                best_param_value)
            df_cv_results.at[exp_param['name'], (self.name, "param")] = best_param_string
            print(df_cv_results.to_latex())
            # print(df_cv_results)
            # print("Best parameters set found on development set:")
            # print()
            # print(clf.best_params_)
            # print()
            # print("Grid scores on development set:")
            # print()
            #
            # means = clf.cv_results_['mean_test_score']
            # stds = clf.cv_results_['std_test_score']
            # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            #     print("%0.3f (+/-%0.03f) for %r"
            #           % (mean, std * 2, params))
            #
            # print(clf.best_estimator_.get_booster().get_dump()[0])


experiment_param = [
    {'name': 'k bin GLM', 'search parameter name': 'Number of bins',
     'xgb_params': {'booster': 'gbdtl', 'discrete_transform': 'identity', 'max_coefficients': 64},
     'search parameter': {'max_bin': range(2, 65, bin_step_size)}},
    {'name': 'DTL.dct', 'search parameter name': 'Number of coefficients',
     'xgb_params': {'booster': 'gbdtl', 'discrete_transform': 'dct'},
     'search parameter': {'max_coefficients': range(2, 65, bin_step_size)}},
    {'name': 'DTL.haar', 'search parameter name': 'Number of coefficients',
     'xgb_params': {'booster': 'gbdtl', 'discrete_transform': 'haar'},
     'search parameter': {'max_coefficients': range(2, 65, bin_step_size)}},
    {'name': '64 bin GLM - 1asso', 'search parameter name': 'Normalised l1 penalty',
     'xgb_params': {'booster': 'gbdtl', 'max_coefficients': 64, 'discrete_transform': 'identity'},
     'search parameter': {'reg_alpha': l1_range}},
    {'name': 'DTL.dct - lasso', 'search parameter name': 'Normalised l1 penalty',
     'xgb_params': {'booster': 'gbdtl', 'discrete_transform': 'dct', 'max_coefficients': 64},
     'search parameter': {'reg_alpha': l1_range}},
    {'name': 'DTL.haar - lasso', 'search parameter name': 'Normalised l1 penalty',
     'xgb_params': {'booster': 'gbdtl', 'discrete_transform': 'haar', 'max_coefficients': 64},
     'search parameter': {'reg_alpha': l1_range}},
]

experiments = [
    Experiment("Higgs", "binary:logistic", "error", data_loader.get_higgs, experiment_param),
    Experiment("Synthetic classification", "binary:logistic", "error", data_loader.get_synthetic_classification,
               experiment_param, 10000),
    Experiment("Cover type", "multi:softmax", "merror", data_loader.get_cover_type, experiment_param),
    Experiment("YearPredictMSD", "reg:linear", "rmse", data_loader.get_year, experiment_param),
    Experiment("Adult", "binary:logistic", "error", data_loader.get_adult, experiment_param),
    Experiment("Wine Quality", "reg:linear", "rmse", data_loader.get_wine_quality, experiment_param),
]

df_description = pd.DataFrame(columns=["Algorithm", "Grid search parameter", "Parameter range"])
for p in experiment_param:
    range_string = str(list(p["search parameter"].values())[0][0]) + "-" + str(
        str(list(p["search parameter"].values())[0][-1]) + str())
    df_description = df_description.append(
        {"Algorithm": p["name"], "Grid search parameter": p["search parameter name"], "Parameter range": range_string},
        ignore_index=True)
column_tuples = []
for e in experiments:
    column_tuples.append((e.name, "param"))
    column_tuples.append((e.name, e.metric))
df_cv_results = pd.DataFrame(columns=pd.MultiIndex.from_tuples(column_tuples, names=['Dataset', '']))
for exp in experiments:
    exp.run()

print(df_description.to_latex(index=False))
print(df_cv_results.to_latex())
