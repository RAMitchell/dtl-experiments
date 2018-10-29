import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import KBinsDiscretizer
import xgboost as xgb
from sklearn.model_selection import train_test_split
import ml_dataset_loader.datasets as data_loader
import pandas as pd


def discretize(X, max_bins):
    enc = KBinsDiscretizer(n_bins=max_bins, encode='onehot-dense')
    enc.fit(X)
    enc.bin_edges_ = [np.unique(edges) for edges in enc.bin_edges_]
    return enc.transform(X)


df = pd.DataFrame(columns=['Algorithm', 'Grid search parameter', 'Best parameter', 'Test RMSE'])
df['Algorithm'] = df['Algorithm'].astype(str)

test_size = 0.2
random_seed = 7
X, y = data_loader.get_wine_quality()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    random_state=random_seed)
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

max_bins = 64
X_binned = discretize(X, max_bins)

X_train_binned, X_test_binned, y_train_binned, y_test_binned = train_test_split(X_binned, y,
                                                                                test_size=test_size,
                                                                                random_state=random_seed)
# LassoCV
lasso_model = LassoCV(cv=5, random_state=0).fit(X_train_binned, y_train_binned)
lasso_metric = np.sqrt(mean_squared_error(y_test_binned, lasso_model.predict(X_test_binned)))
df = df.append({'Algorithm': '64 bin Lasso', 'Grid search parameter': 'normalised l1 penalty',
                'Best parameter': str(np.round(lasso_model.alpha_, 4)), 'Test RMSE': lasso_metric}, ignore_index=True)

# ElasticNetCV
en_model = ElasticNetCV(cv=5, random_state=0).fit(X_train_binned, y_train_binned)
en_metric = np.sqrt(mean_squared_error(y_test_binned, en_model.predict(X_test_binned)))
df = df.append({'Algorithm': '64 bin Elastic Net (0.5 l1 ratio)', 'Grid search parameter': 'normalised l1/l2 penalty',
                'Best parameter': str(np.round(en_model.alpha_, 4)), 'Test RMSE': en_metric}, ignore_index=True)


bins = range(2, 32)
scores = []
for bin in bins:
    dtl_model = xgb.XGBRegressor(n_estimators=200, booster='gbdtl', max_bin=bin, discrete_transform='identity',
                                 max_coefficients=bin)
    cv_scores = cross_val_score(dtl_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    scores.append(cv_scores.mean())
best_bin = bins[np.argmax(scores)]

dtl_model = xgb.XGBRegressor(n_estimators=200, booster='gbdtl', max_bin=best_bin, discrete_transform='identity',
                             max_coefficients=best_bin)
dtl_model.fit(X_train, y_train)
cv_bin_metric = np.sqrt(mean_squared_error(y_test, dtl_model.predict(X_test)))
df = df.append(
    {'Algorithm': 'k-bin Linear Regression', 'Grid search parameter': 'Number of bins', 'Best parameter': str(best_bin),
     'Test RMSE': cv_bin_metric}, ignore_index=True)

# dct
dct_coefficients = range(2, 16)
scores = []
for coefficient in dct_coefficients:
    dtl_model = xgb.XGBRegressor(n_estimators=200, booster='gbdtl', max_bin=64, discrete_transform='dct',
                                 max_coefficients=coefficient)
    cv_scores = cross_val_score(dtl_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    scores.append(cv_scores.mean())
best_coefficients = dct_coefficients[np.argmax(scores)]

dtl_model = xgb.XGBRegressor(n_estimators=200, booster='gbdtl', max_coefficients=best_coefficients)
dtl_model.fit(X_train, y_train)
dtl_metric = np.sqrt(mean_squared_error(y_test, dtl_model.predict(X_test)))
df = df.append({'Algorithm': '64 bin DTL with DCT', 'Grid search parameter': 'DCT coefficients',
                'Best parameter': str(best_coefficients), 'Test RMSE': dtl_metric}, ignore_index=True)

# haar
haar_coefficients = range(2, 16)
scores = []
for coefficient in haar_coefficients:
    dtl_model = xgb.XGBRegressor(n_estimators=200, booster='gbdtl', max_bin=64, discrete_transform='haar',
                                 max_coefficients=coefficient)
    cv_scores = cross_val_score(dtl_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    scores.append(cv_scores.mean())
best_coefficients = dct_coefficients[np.argmax(scores)]

dtl_model = xgb.XGBRegressor(n_estimators=200, booster='gbdtl', max_bin=64, discrete_transform='haar',
                             max_coefficients=best_coefficients)
dtl_model.fit(X_train, y_train)
dtl_metric = np.sqrt(mean_squared_error(y_test, dtl_model.predict(X_test)))
df = df.append({'Algorithm': '64 bin DTL with Haar', 'Grid search parameter': 'Haar coefficients',
                'Best parameter': str(best_coefficients), 'Test RMSE': dtl_metric}, ignore_index=True)

print(df.to_latex(index=False))
