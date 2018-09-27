import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os


def rmse(y, pred):
    metric = np.sqrt(mean_squared_error(y, pred))
    return "\(rmse = %s\)" % float('%.4g' % metric)


figures_dir = "figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],  # use latex default serif font
})

random_seed = 7
np.random.seed(random_seed)
num_rounds = 100
num_stumps = [8, 32]
res = {}
n_train = 50
n_test = 500
X_train = np.zeros((n_train, 1))
X_train[:, 0] = np.linspace(-3.0, 3.0, n_train)
X_test = np.zeros((n_test, 1))
X_test[:, 0] = np.linspace(-3.0, 3.0, n_test)
y_true = [x ** 5 - 8 * x ** 3 + 10 * x + 6 for x in X_train[:, 0]]
y_test = [x ** 5 - 8 * x ** 3 + 10 * x + 6 for x in X_test[:, 0]]
y_noisy = [y + np.random.normal(scale=10) for y in y_true]
dtrain = xgb.DMatrix(X_train, y_noisy)
dtest = xgb.DMatrix(X_test, y_test)
true_label = "$f(x)=x^{5}-8x^{3}+10x+6$"
noisy_label = "$y_{train}=f(x)+\epsilon$"
k = 8
dct_k = k
param = {"booster": "gbdct", "max_coefficients": dct_k,"max_bin":k}
bst = xgb.train(param, dtrain, num_rounds)
train_pred = bst.predict(dtrain)
test_pred = bst.predict(dtest)
plt.plot(X_train[:, 0], train_pred, label="step function, \(k=" + str(k) + "\), " + rmse(y_test, test_pred))

plt.plot(X_train[:, 0], y_noisy, ".", label=noisy_label)
plt.plot(X_train[:, 0], y_true, label=true_label)

plt.xlabel('x')
plt.legend()
plt.savefig(figures_dir + '/' + 'step_k_8.pgf', bbox_inches='tight')
plt.savefig(figures_dir + '/' + 'step_k_8.png', bbox_inches='tight')
plt.clf()

k = 32
dct_k = k
param = {"booster": "gbdct", "max_coefficients": dct_k,"max_bin":k}
bst = xgb.train(param, dtrain, num_rounds)
train_pred = bst.predict(dtrain)
test_pred = bst.predict(dtest)
plt.plot(X_train[:, 0], train_pred, label="step function, \(k=" + str(k) + "\), " + rmse(y_test, test_pred))

plt.plot(X_train[:, 0], y_noisy, ".", label=noisy_label)
plt.plot(X_train[:, 0], y_true, label=true_label)

plt.xlabel('x')
plt.legend()
plt.savefig(figures_dir + '/' + 'step_k_32.pgf', bbox_inches='tight')
plt.savefig(figures_dir + '/' + 'step_k_32.png', bbox_inches='tight')
plt.clf()

k = 32
dct_k = 8
param = {"booster": "gbdct", "max_coefficients": dct_k,"max_bin":k}
bst = xgb.train(param, dtrain, num_rounds)
train_pred = bst.predict(dtrain)
test_pred = bst.predict(dtest)
plt.plot(X_train[:, 0], train_pred, label="dct function, \(k=" + str(k) + "\), \(dct\_k=" + str(dct_k) + "\), " + rmse(y_test, test_pred))

plt.plot(X_train[:, 0], y_noisy, ".", label=noisy_label)
plt.plot(X_train[:, 0], y_true, label=true_label)

plt.xlabel('x')
plt.legend()
plt.savefig(figures_dir + '/' + 'dct_8_32.pgf', bbox_inches='tight')
plt.savefig(figures_dir + '/' + 'dct_8_32.png', bbox_inches='tight')
plt.clf()
