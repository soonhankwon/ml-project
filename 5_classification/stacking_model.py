from multiprocessing.spawn import import_main_path
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer_data = load_breast_cancer()

X_data = cancer_data.data
y_label = cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.2, random_state=0)

# 개별 ML 모델을 위한 Classifier 생성
knn_clf = KNeighborsClassifier(n_neighbors=4)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
dt_clf = DecisionTreeClassifier()
ada_clf = AdaBoostClassifier(n_estimators=100)

# 최종 Stacking 모델을 위한 Classifier 생성
lr_final = LogisticRegression(C=10)

# 개별 모델들을 학습
knn_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
dt_clf.fit(X_train, y_train)
ada_clf.fit(X_train, y_train)

# 학습된 개별 모델들이 각자 반환하는 예측 데이터셋을 생성하고 개별 모델의 정확도 측정
knn_pred = knn_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
dt_pred = dt_clf.predict(X_test)
ada_pred = ada_clf.predict(X_test)

print(f'KNN 정확도: {accuracy_score(y_test, knn_pred):.4f}')
print(f'랜덤 포레스트 정확도: {accuracy_score(y_test, rf_pred):.4f}')
print(f'결정트리 정확도: {accuracy_score(y_test, dt_pred):.4f}')
print(f'Ada Boost 정확도: {accuracy_score(y_test, ada_pred):.4f}')
"""
KNN 정확도: 0.9211
랜덤 포레스트 정확도: 0.9649
결정트리 정확도: 0.9123
Ada Boost 정확도: 0.9737
"""

# 개별 모델의 예측결과를 메타모델이 학습할 수 있도록 스태킹 형태로 재생성
pred = np.array([knn_pred, rf_pred, dt_pred, ada_pred])
print(pred.shape)
"""
(4, 114)
"""

# transpose를 이용해 행과 열의 위치 교환, 컬럼 레벨로 각 알고리즘의 예측 결과를 피처로 만듬
pred = np.transpose(pred)
print(pred.shape)
"""
(114, 4)
"""

# 메타모델 학습/예측/평가
lr_final.fit(pred, y_test) # overfitting
final = lr_final.predict(pred)

print(f'최종 메타 모델의 예측 정확도: {accuracy_score(y_test, final):.4f}')
"""
최종 메타 모델의 예측 정확도: 0.9737
"""
