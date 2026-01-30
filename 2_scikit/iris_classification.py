from pandas.core.common import random_state
import sklearn
print(sklearn.__version__)

# 붓꽃 예측을 위한 사이킷런 필요 모듈 로딩
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# 붓꽃 데이터 세트 로딩
iris = load_iris()

# iris.data는 iris 데이터 세트에서 피처(feature)만으로 된 데이터를 numpy로 가지고 있음
iris_data = iris.data

# iris.target은 붓꽃 데이터 세트에서 레이블(결정 값) 데이터를 numpy로 가지고 있음
iris_label = iris.target
print('iris target 값:', iris_label)
print('iris target 명:', iris.target_names)
"""
iris target 값: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
iris target 명: ['setosa' 'versicolor' 'virginica']
"""

# 붓꽃 데이터 세트를 DataFrame으로 변환
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df.head(3))
"""
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  label
0                5.1               3.5                1.4               0.2      0
1                4.9               3.0                1.4               0.2      0
2                4.7               3.2                1.3               0.2      0
"""

# 학습 데이터와 테스트 데이터 세트로 분리(X: 피처, y:타겟)
X_train, X_test, y_train, y_test = train_test_split(
    iris_data, iris_label, test_size=0.2, random_state=11)

# 학습 데이터 세트로 학습(Train) 수행
# DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)

# 학습 수행
dt_clf.fit(X_train, y_train)

# 테스트 데이터 세트로 예측(Predict) 수행
pred = dt_clf.predict(X_test) # 테스트시 문제만 내줌(피처 데이터셋)
print(pred)
"""
[2 2 1 1 2 0 1 0 0 1 1 1 1 2 2 0 2 1 2 2 1 0 0 1 0 0 2 1 0 1]
"""

# 예측 정확도 평가
from sklearn.metrics import accuracy_score
print('예측 정확도: {0: 4f}'.format(accuracy_score(y_test, pred)))
"""
예측 정확도:  0.933333
"""
