import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import ml_dataset_loader.datasets as data_loader
import matplotlib
from sklearn.ensemble import ExtraTreesClassifier

feature_idx  = 0
X, y = data_loader.get_adult()
dtl_model = xgb.XGBClassifier(n_estimators=200, booster='gbdtl',max_coefficients=8)
dtl_model.fit(X,y)
min_age = X[:,feature_idx].min()
max_age = X[:,feature_idx].max()

m =1000
X_test = np.ndarray((m,X.shape[1]))
X_test.fill(np.nan)
X_test[:,feature_idx] = np.linspace(min_age,max_age, m)
pred = dtl_model.predict_proba(X_test)

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],  # use latex default serif font
})
plt.plot(X_test[:,feature_idx],pred[:,1] )
plt.xlabel('Age')
plt.ylabel('Probability Income$>$50K')
plt.savefig('figures/interpretability_age.pgf', bbox_inches='tight')
plt.savefig('figures/interpretability_age.png', bbox_inches='tight')

feature_idx  = 5
X, y = data_loader.get_adult()
dtl_model = xgb.XGBClassifier(n_estimators=200, booster='gbdtl',max_coefficients=8)
dtl_model.fit(X,y)
min_age = X[:,feature_idx].min()
max_age = X[:,feature_idx].max()

m =1000
X_test = np.ndarray((m,X.shape[1]))
X_test.fill(np.nan)
X_test[:,feature_idx] = np.linspace(min_age,max_age, m)
pred = dtl_model.predict_proba(X_test)

plt.clf()
plt.plot(X_test[:,feature_idx],pred[:,1] )
plt.xlabel('Hours per week')
plt.ylabel('Probability Income$>$50K')
plt.savefig('figures/interpretability_hours.pgf', bbox_inches='tight')
plt.savefig('figures/interpretability_hours.png', bbox_inches='tight')
