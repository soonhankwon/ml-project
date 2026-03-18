from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np

diabetes = load_diabetes()
X_train, X_val, y_train, y_val = train_test_split(diabetes.data, diabetes.target, random_state=0)

# 학습, 예측, R2 Score 평가
model = Ridge(alpha=1e-2).fit(X_train, y_train)
y_pred = model.predict(X_val)
print('r2 score:', r2_score(y_val, y_pred))
"""
r2 score: 0.35666753229394244
"""

from sklearn.inspection import permutation_importance

r = permutation_importance(model, X_val, y_val, n_repeats=30, random_state=0)

# 가장 평균 permutation importance가 높은 순으로 내림차순 정렬 후 평균 permutation importance값과 표준 편차 출력
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(diabetes.feature_names[i], " ", np.round(r.importances_mean[i], 4), " +/-", np.round(r.importances_std[i], 5))

"""
s5   0.2042  +/- 0.04964
bmi   0.1758  +/- 0.0484
bp   0.0884  +/- 0.03284
sex   0.0559  +/- 0.02319
"""

