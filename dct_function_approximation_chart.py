import xgboost as xgb
import os
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler


def rmse(y, pred):
    metric = np.sqrt(mean_squared_error(y, pred))
    return "rmse: %s" % float('%.4g' % metric)

figures_dir = "figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],  # use latex default serif font
})

random_seed = 7
num_rounds = 5
num_stumps = [32]
res = {}
n = 1000
X = np.zeros((n, 1))
X[:, 0] = np.linspace(-3.0, 3.0, n)
y = [x ** 5 - 8 * x ** 3 + 10 * x + 6 for x in X[:, 0]]
dct_coefficients = [4, 8, 16]
plt.rc('axes', prop_cycle=(cycler('color', ['g', 'r', 'b', 'y', 'c', 'm']) +
                           cycler('linestyle', ['-', '-', '-', '-', '-', '-'])))
plt.plot(X[:, 0], y,label="$y=x^5-8x^3+10x+6$")
'''
for max_coefficients in dct_coefficients:
    param = {"booster": "gbdtl", "max_coefficients": max_coefficients, "max_bins":32}
    dtrain = xgb.DMatrix(X, y)
    bst = xgb.train(param, dtrain, num_rounds)
    dct_pred = bst.predict(dtrain)
    plt.plot(X[:, 0], dct_pred, label=str(max_coefficients) + "/32 dct coefficients, " + rmse(y,dct_pred))
'''
haar_coefficients = [(4,8),(16,128)]
for coefficients in haar_coefficients:
    param = {"booster": "gbdtl", "max_coefficients": coefficients[0], "max_bin": coefficients[1], "discrete_transform":"haar"}
    dtrain = xgb.DMatrix(X, y)
    bst = xgb.train(param, dtrain, num_rounds)
    dct_pred = bst.predict(dtrain)
    plt.plot(X[:, 0], dct_pred, label=str(coefficients[0]) + "/" + str(param["max_bin"]) + " haar coefficients, " + rmse(y,dct_pred))
identity_coefficients = [(4,4),(16,16)]
for coefficients in identity_coefficients:
    param = {"booster": "gbdtl", "max_coefficients": coefficients[0], "max_bin": coefficients[1], "discrete_transform":"haar"}
    dtrain = xgb.DMatrix(X, y)
    bst = xgb.train(param, dtrain, num_rounds)
    dct_pred = bst.predict(dtrain)
    plt.plot(X[:, 0], dct_pred, label=str(coefficients[0]) + "/" + str(param["max_bin"]) + " identity coefficients, " + rmse(y,dct_pred))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig(figures_dir + '/' + 'function_approximation1.pgf', bbox_inches='tight')
plt.savefig(figures_dir + '/' + 'function_approximation1.png', bbox_inches='tight')
'''
plt.clf()

plt.rc('axes', prop_cycle=(cycler('color', ['g', 'y', 'c', 'm', 'b', 'r']) +
                           cycler('linestyle', ['-', '-', '-', '-', '-', '-'])))
plt.plot(X[:, 0], y,label="$y=x^5-8x^3+10x+6$")

param = {"booster": "gbdtl", "max_coefficients": 16, "max_bins":32}
dtrain = xgb.DMatrix(X, y)
bst = xgb.train(param, dtrain, num_rounds)
dct_pred = bst.predict(dtrain)
plt.plot(X[:, 0], dct_pred, label=str(max_coefficients) + "/32 dct coefficients, " + rmse(y,dct_pred))

param = {"booster": "gbdtl", "max_coefficients": 16, "max_bins":32, "discrete_transform":"haar"}
dtrain = xgb.DMatrix(X, y)
bst = xgb.train(param, dtrain, num_rounds)
dct_pred = bst.predict(dtrain)
plt.plot(X[:, 0], dct_pred, label=str(max_coefficients) + "/32 haar coefficients, " + rmse(y,dct_pred))

param = {"booster": "gbtree", "max_depth": 1, "reg_lambda":0}
for stumps in num_stumps:
    bst = xgb.train(param, dtrain, stumps)
    stump_pred = bst.predict(dtrain)
    metric = rmse(y, stump_pred)
    plt.plot(X[:, 0], stump_pred, label=str(stumps) + " decision stumps, "+ rmse(y,stump_pred))

param = {"booster": "gblinear"}
bst = xgb.train(param, dtrain, num_rounds)
linear_pred = bst.predict(dtrain)
metric = rmse(y, linear_pred)
plt.plot(X[:, 0], linear_pred, label="linear, "+ rmse(y,linear_pred))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig(figures_dir + '/' + 'function_approximation2.pgf', bbox_inches='tight')
plt.savefig(figures_dir + '/' + 'function_approximation2.png', bbox_inches='tight')
plt.clf()
'''

