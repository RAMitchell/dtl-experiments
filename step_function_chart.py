import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.stats import binned_statistic


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
"""
k = 8
dct_k = k
param = {"booster": "gbdtl", "max_coefficients": dct_k,"max_bin":k, "discrete_transform":"identity"}
bst = xgb.train(param, dtrain, num_rounds)
train_pred = bst.predict(dtrain)
test_pred = bst.predict(dtest)
plt.plot(X_test[:, 0], test_pred, label="step function, \(k=" + str(k) + "\), " + rmse(y_test, test_pred))

plt.plot(X_train[:, 0], y_noisy, ".", label=noisy_label)
plt.plot(X_train[:, 0], y_true, label=true_label)

plt.xlabel('x')
plt.legend()
plt.savefig(figures_dir + '/' + 'step_k_8.pgf', bbox_inches='tight')
plt.savefig(figures_dir + '/' + 'step_k_8.png', bbox_inches='tight')
plt.clf()

k = 32
dct_k = k
param = {"booster": "gbdtl", "max_coefficients": dct_k,"max_bin":k, "discrete_transform":"identity"}
bst = xgb.train(param, dtrain, num_rounds)
train_pred = bst.predict(dtrain)
test_pred = bst.predict(dtest)
plt.plot(X_test[:, 0], test_pred, label="step function, \(k=" + str(k) + "\), " + rmse(y_test, test_pred))

plt.plot(X_train[:, 0], y_noisy, ".", label=noisy_label)
plt.plot(X_train[:, 0], y_true, label=true_label)

plt.xlabel('x')
plt.legend()
plt.savefig(figures_dir + '/' + 'step_k_32.pgf', bbox_inches='tight')
plt.savefig(figures_dir + '/' + 'step_k_32.png', bbox_inches='tight')
plt.clf()

k = 32
dct_k = 8
param = {"booster": "gbdtl", "max_coefficients": dct_k,"max_bin":k}
bst = xgb.train(param, dtrain, num_rounds)
train_pred = bst.predict(dtrain)
test_pred = bst.predict(dtest)
plt.plot(X_test[:, 0], test_pred, label="dct function, \(k=" + str(k) + "\), \(dct\_k=" + str(dct_k) + "\), " + rmse(y_test, test_pred))

plt.plot(X_train[:, 0], y_noisy, ".", label=noisy_label)
plt.plot(X_train[:, 0], y_true, label=true_label)

plt.xlabel('x')
plt.legend()
plt.savefig(figures_dir + '/' + 'dct_8_32.pgf', bbox_inches='tight')
plt.savefig(figures_dir + '/' + 'dct_8_32.png', bbox_inches='tight')
plt.clf()

from scipy.fftpack import dct
bin_means = binned_statistic(X_train[:, 0], y_true,bins=32)
y_dct = dct(bin_means[0],norm="ortho")
bin_means = binned_statistic(X_train[:, 0], np.subtract(y_noisy,y_true),bins=32)
y_noise_only_dct = dct(bin_means[0], norm="ortho")
bar_width = 0.4
plt.bar(np.arange(len(y_dct)),np.square(y_dct),bar_width ,label=true_label)
plt.bar(np.arange(len(y_dct))+bar_width, np.square(y_noise_only_dct),bar_width, label="$\epsilon$")
plt.xlabel('DCT coefficient')
plt.ylabel('Spectral power')
plt.legend()
plt.savefig(figures_dir + '/' + 'dct_spectral_power.pgf', bbox_inches='tight')
plt.savefig(figures_dir + '/' + 'dct_spectral_power.png', bbox_inches='tight')
plt.clf()
"""


def haar_matrix(m, n):
    T = np.ndarray((m, n))
    scale = 1.0 / np.sqrt(float(m))
    for j in range(n):
        p = np.floor(np.log2(float(j)))
        exp_p = np.power(2.0, p)
        exp_p_half = np.power(2.0, p / 2.0)
        q = (j - exp_p) + 1.0
        for i in range(n):
            t = float(i) / m
            if j == 0:
                T[i, j] = scale
            elif (q - 1.0) / exp_p <= t and t < (q - 0.5) / exp_p:
                T[i, j] = scale * exp_p_half
            elif (q - 0.5) / exp_p <= t and t < q / exp_p:
                T[i, j] = scale * -exp_p_half
            else:
                T[i, j] = 0
    return T


T = haar_matrix(32,32)
bin_means = binned_statistic(X_train[:, 0], y_true, bins=32)
y_dct = T.T.dot(bin_means[0])
bin_means = binned_statistic(X_train[:, 0], np.subtract(y_noisy, y_true), bins=32)
y_noise_only_haar = T.dot(bin_means[0])
bar_width = 0.4
plt.bar(np.arange(len(y_dct)), np.square(y_dct), bar_width, label=true_label)
plt.bar(np.arange(len(y_dct)) + bar_width, np.square(y_noise_only_haar), bar_width, label="$\epsilon$")
plt.xlabel('Haar coefficient')
plt.ylabel('Spectral power')
plt.legend()
plt.show()
plt.savefig(figures_dir + '/' + 'haar_spectral_power.pgf', bbox_inches='tight')
plt.savefig(figures_dir + '/' + 'haar_spectral_power.png', bbox_inches='tight')
plt.clf()
