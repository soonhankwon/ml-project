import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# StandardScaler()로 평균이 0, 분산 1로 데이터 분포도 변환
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)

X_train, X_test, y_train, y_test = train_test_split(data_scaled, cancer.target, test_size=0.3, random_state=0)

from sklearn.metrics import accuracy_score, roc_auc_score

# 로지스틱 회귀를 이용하여 학습 및 예측 수행
# solver 인자값을 생성자로 입력하지 않으면 solver='lbfgs'
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_preds = lr_clf.predict(X_test)
lr_preds_proba = lr_clf.predict_log_proba(X_test)[:, 1]

# accuracy, roc_auc 측정
print(f'accuracy: {accuracy_score(y_test, lr_preds):.3f} roc_auc:{roc_auc_score(y_test, lr_preds_proba):.3f}')

"""
accuracy: 0.977 roc_auc:0.995
"""

solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
# 여러개의 solver값 별로 LogisticRegression 학습 후 성능 평가
for solver in solvers:
    lr_clf = LogisticRegression(solver=solver, max_iter=600)
    lr_clf.fit(X_train, y_train)
    lr_preds = lr_clf.predict(X_test)
    lr_preds_proba = lr_clf.predict_proba(X_test)[:, 1]

    # accuracy와 roc_auc 측정
    print(f'solver:{solver}, accuracy: {accuracy_score(y_test, lr_preds):.3f}, roc_auc:{roc_auc_score(y_test , lr_preds_proba):.3f}')

"""
solver:lbfgs, accuracy: 0.977, roc_auc:0.995
solver:liblinear, accuracy: 0.982, roc_auc:0.995
solver:newton-cg, accuracy: 0.977, roc_auc:0.995
solver:sag, accuracy: 0.982, roc_auc:0.995
solver:saga, accuracy: 0.982, roc_auc:0.995
"""

from sklearn.model_selection import GridSearchCV

# sklearn 1.8+: penalty 대신 l1_ratio (0=L2, 1=L1). lbfgs는 L2만 지원.
param_grid = [
    {'solver': ['lbfgs'], 'l1_ratio': [0], 'C': [0.01, 0.1, 1, 5, 10]},
    {'solver': ['liblinear'], 'l1_ratio': [0, 1], 'C': [0.01, 0.1, 1, 5, 10]},
]

lr_clf = LogisticRegression()

grid_clf = GridSearchCV(lr_clf, param_grid=param_grid, scoring='accuracy', cv=3)
grid_clf.fit(data_scaled, cancer.target)
print(f'최적 하이퍼 파라미터:{grid_clf.best_params_}, 최적 평균 정확도:{grid_clf.best_score_:.3f}')
"""
최적 하이퍼 파라미터:{'C': 0.1, 'l1_ratio': 0, 'solver': 'liblinear'}, 최적 평균 정확도:0.979
"""