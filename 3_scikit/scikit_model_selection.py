from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target
# dt_clf.fit(train_data, train_label)

# # 학습 데이터 셋으로 예측 수행
# pred = dt_clf.predict(train_data)
# print('예측 정확도:', accuracy_score(train_label, pred))
# 예측 정확도: 1.0(학습 데이터로 예측을 수행했기 때문에)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=121)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print(f'예측 정확도: {accuracy_score(y_test, pred):.4f}')
# 예측 정확도: 0.9556

# 넘파이 ndarray 뿐만 아니라 판다스 DataFrame/Series도 train_test_split()으로 분할 가능

import pandas as pd

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
print(iris_df.head())

ftr_df = iris_df.iloc[:, :-1] # 특성 4개 컬럼(마지막 열 제외, slice)
tgt_df = iris_df.iloc[:, -1] # 마지막 열만(target)
X_train, X_test, y_train, y_test = train_test_split(ftr_df, tgt_df, test_size=0.3, random_state=121)

print(type(X_train), type(X_test), type(y_train), type(y_test))

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print(f'예측 정확도: {accuracy_score(y_test, pred):.4f}')

"""
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
2                4.7               3.2                1.3               0.2       0
3                4.6               3.1                1.5               0.2       0
4                5.0               3.6                1.4               0.2       0
<class 'pandas.DataFrame'> <class 'pandas.DataFrame'> <class 'pandas.Series'> <class 'pandas.Series'>
예측 정확도: 0.9556
"""
