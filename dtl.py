import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from cycler import cycler

random_seed = 7
num_rounds =200
res = {}
n = 10000
#X, y = make_classification(n_samples=n, random_state=random_seed, n_features=2,n_informative=2, n_redundant=0)
X, y = make_classification(n_samples=n, random_state=random_seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)
dct_coefficients = [8, 16]
common_param = {"objective": "binary:logistic","reg_lambda":0,"max_bin":64}
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'r', 'g', 'g', 'b', 'b', 'y', 'y','m','m','k','k']) +
                           cycler('linestyle', ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--'])))

for max_coefficients in dct_coefficients:
    param = {"booster": "gbdct", "max_coefficients": max_coefficients,"regularising_transform":"dct"}
    param.update(common_param)
    res = {}
    bst = xgb.train(param, dtrain, num_rounds, evals=[(dtrain, "train"), (dtest, "test")], evals_result=res)
    plt.plot(res["train"]["error"], label=str(max_coefficients) + " " + param["regularising_transform"] + " coefficients - train")
    plt.plot(res["test"]["error"], label=str(max_coefficients) + " " + param["regularising_transform"] + " coefficients - test")

identity_coefficients = [8, 16]
for max_coefficients in dct_coefficients:
    param = {"booster": "gbdct", "max_coefficients": max_coefficients,"regularising_transform":"haar"}
    param.update(common_param)
    #param["max_bin"] = max_coefficients
    res = {}
    bst = xgb.train(param, dtrain, num_rounds, evals=[(dtrain, "train"), (dtest, "test")], evals_result=res)
    plt.plot(res["train"]["error"], label=str(max_coefficients) + " " + param["regularising_transform"] + " coefficients - train")
    plt.plot(res["test"]["error"], label=str(max_coefficients) + " " + param["regularising_transform"] + " coefficients - test")

param = {"booster": "gblinear"}
param.update(common_param)
res = {}
bst = xgb.train(param, dtrain, num_rounds, evals=[(dtrain, "train"), (dtest, "test")], evals_result=res)
plt.plot(res["train"]["error"], label="linear model - train")
plt.plot(res["test"]["error"], label="linear model - test")

param = {"booster": "gbtree","max_depth":1}
param.update(common_param)
bst = xgb.train(param, dtrain, num_rounds, evals=[(dtrain, "train"), (dtest, "test")], evals_result=res)
plt.plot(res["train"]["error"], label="decision stumps - train")
plt.plot(res["test"]["error"], label="decision stumps - test")
plt.xlabel('iterations')
plt.ylabel('error')
plt.legend()
plt.show()
