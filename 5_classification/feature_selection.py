import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from sklearn.datasets import make_classification

# Recursive Feature Elimination
# 분류를 위한 Feature 개수가 25개인 데이터 1000개 생성
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
n_redundant=2, n_repeated=0, n_classes=8, n_clusters_per_class=1, random_state=0)

# SVC classifier 선택
svc = SVC(kernel='linear')
# REFC로 Feature들을 반복적으로 제거하면서 학습/평가 수행
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy', verbose=2)
rfecv.fit(X, y)

print('Optimal number of features: %d' %rfecv.n_features_)
"""
Optimal number of features: 3
"""
# Plot number of features VS cross-validation scores
plt.figure()
plt.xlabel('Number of features selected')
plt.ylabel('Cross validation score (nb of correct classifications)')
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.show()

# SelectFromModel
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
print(diabetes.DESCR)

import numpy as np
from sklearn.linear_model import LassoCV

lasso = LassoCV().fit(X, y)
importance = np.abs(lasso.coef_)
feature_names = np.array(diabetes.feature_names)
plt.bar(height=importance, x=feature_names)
plt.title('Feature importance via coefficients')
plt.show()

from sklearn.feature_selection import SelectFromModel
from time import time

threshold = np.sqrt(importance)[-3] + 0.01
print('threshold:', threshold)
sfm = SelectFromModel(lasso, threshold=threshold).fit(X, y)
print(f'Features selected by SelectFromModel: {feature_names[sfm.get_support()]}')
"""
threshold: 11.99743153238703
Features selected by SelectFromModel: ['sex' 'bmi' 'bp' 's1' 's2' 's4' 's5' 's6']
"""

