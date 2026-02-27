import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# 위스콘신 유방암 데이터셋 로드
cancer = load_breast_cancer()

data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
print(data_df.head(3))
"""
   mean radius  mean texture  mean perimeter  ...  worst concave points  worst symmetry  worst fractal dimension
0        17.99         10.38           122.8  ...                0.2654          0.4601                  0.11890
1        20.57         17.77           132.9  ...                0.1860          0.2750                  0.08902
2        19.69         21.25           130.0  ...                0.2430          0.3613                  0.08758

[3 rows x 30 columns]
"""

# VotingClasifier로 개별모델은 로지스틱 회귀와 KNN 보팅방식으로 결합하고 성능 비교
lr_clf = LogisticRegression(max_iter=10000)
knn_clf = KNeighborsClassifier(n_neighbors=8)

# 개별 모델을 소프트보팅 기반의 앙상블 모델로 구현한 분류기
vo_clf = VotingClassifier(estimators=[('LR', lr_clf), ('KNN', knn_clf)], voting='soft')
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)

# VotingClassifier 학습/예측/평가
vo_clf.fit(X_train, y_train)
pred = vo_clf.predict(X_test)
print(f'Voting 분류기 정확도: {accuracy_score(y_test, pred):.4f}')

# 개별 모델의 학습/예측/평가
classifiers = [lr_clf, knn_clf]
for clasifier in classifiers:
    clasifier.fit(X_train, y_train)
    pred = clasifier.predict(X_test)
    class_name = clasifier.__class__.__name__
    print(f'{class_name} 정확도: {accuracy_score(y_test, pred):.4f}')

"""
Voting 분류기 정확도: 0.9474
LogisticRegression 정확도: 0.9649
KNeighborsClassifier 정확도: 0.9386
"""

