from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import numpy as np

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target

# 성능 지표는 정확도(accuracy), 교차 검증 세트는 3개 - Stratified K-fold
scores = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=3)

# print(scores, type(scores))
print('교차 검증별 정확도: ', np.round(scores, 4))
print('평균 검증 정확도: ', np.round(np.mean(scores), 4))
"""
교차 검증별 정확도:  [0.98 0.94 0.98]
평균 검증 정확도:  0.9667
"""

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=121)
dtree = DecisionTreeClassifier() # max_depth, min_samples_split 메서드 파라미터 확인가능

### parameter들을 dictionary 형태로 설정
parameters = {'max_depth': [1,2,3], 'min_samples_split':[2,3]}

import pandas as pd

# param_grid의 하이퍼 파라미터들을 3개의 train, test set fold로 나누어서 테스트 수행 설정
## refit=True가 default, True면 가장 좋은 파라미터 설정으로 재학습
grid_dtree = GridSearchCV(dtree, param_grid=parameters, refit=True, return_train_score=True)

# 붓꽃 Train 데이터로 param_grid 하이퍼 파라미터들을 순차적으로 학습/평가
grid_dtree.fit(X_train, y_train)

# GridSearchCV 결과는 cv_results_ 라는 딕셔너리로 저장됨. 이를 DataFrame으로 변환

scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', \
           'split0_test_score', 'split1_test_score', 'split2_test_score']]
print(scores_df)
"""
   mean_fit_time  std_fit_time  mean_score_time  ...  split4_train_score  mean_train_score  std_train_score
0       0.000181      0.000006         0.000119  ...            0.697917          0.700000         0.004167
1       0.000171      0.000002         0.000111  ...            0.697917          0.700000         0.004167
2       0.000207      0.000029         0.000129  ...            0.968750          0.958333         0.013176
3       0.000182      0.000013         0.000107  ...            0.968750          0.958333         0.013176
4       0.000212      0.000060         0.000121  ...            0.989583          0.977083         0.010206
5       0.000188      0.000012         0.000113  ...            0.989583          0.977083         0.010206
"""

print('GridSearchCV 최적 파라미터:', grid_dtree.best_params_)
print('GridSearchCV 최고 정확도: {0: 4f}'.format(grid_dtree.best_score_))
"""
GridSearchCV 최적 파라미터: {'max_depth': 3, 'min_samples_split': 2}
GridSearchCV 최고 정확도:  0.975000
"""

# refit=True로 설정된 GridSearchCV 객체가 fit()을 수행 시 학습이 완료된 Estimator를 내포하고 있으므로 predict()를 통해 예측도 가능
pred = grid_dtree.predict(X_test)
print('테스트 데이터 세트 정확도: {0: .4f}'.format(accuracy_score(y_test, pred))) 
# 테스트 데이터 세트 정확도:  0.9667 

# GridSearchCV의 refit으로 이미 학습이 된 estimator 반환
estimator = grid_dtree.best_estimator_ 

# GridSearchCV의 best_estimator_ 는 이미 최적 하이퍼 파라미터로 학습이 됨
pred = estimator.predict(X_test)
print('테스트 데이터 세트 정확도: {0: .4f}'.format(accuracy_score(y_test, pred)))
# 테스트 데이터 세트 정확도:  0.9667