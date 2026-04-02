import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
caliDF = pd.DataFrame(housing.data, columns=housing.feature_names)

caliDF['PRICE'] = housing.target
print('California Housing 데이터셋 크기:', caliDF.shape)
print(caliDF.head())
"""
California Housing 데이터셋 크기: (20640, 9)
   MedInc  HouseAge  AveRooms  ...  Latitude  Longitude  PRICE
0  8.3252      41.0  6.984127  ...     37.88    -122.23  4.526
1  8.3014      21.0  6.238137  ...     37.86    -122.22  3.585
2  7.2574      52.0  8.288136  ...     37.85    -122.24  3.521
3  5.6431      52.0  5.817352  ...     37.85    -122.25  3.413
4  3.8462      52.0  6.281853  ...     37.85    -122.25  3.422

[5 rows x 9 columns]
"""

fig, axs = plt.subplots(figsize=(16, 8), ncols=4, nrows=2)
lm_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
for i, feature in enumerate(lm_features):
   row = int(i/4)
   col = i%4
   sns.regplot(x=feature, y='PRICE', data=caliDF, ax=axs[row][col])

# plt.show()

# 학습과 테스트 데이터 세트로 분리하고 학습/예측/평가 수행
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y_target = caliDF['PRICE']
X_data = caliDF.drop(['PRICE'], axis=1, inplace=False)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.3, random_state=156)

# Linear Regression OLS로 학습/예측/평가 수행
lr = LinearRegression()
lr.fit(X_train, y_train)
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse= np.sqrt(mse)

print(f'MSE: {mse:.3f}, RMSE: {rmse:.3f}')
print(f'Variance score: {r2_score(y_test, y_preds):.3f}')
print('절편 값:', lr.intercept_)
print('회귀 계수값:', np.round(lr.coef_, 1))
"""
MSE: 0.543, RMSE: 0.737
Variance score: 0.595
절편 값: -37.23905305294159
회귀 계수값: [ 0.4  0.  -0.1  0.6 -0.  -0.  -0.4 -0.4]
"""

# 회귀 계수를 큰 값 순으로 정렬하기 위해 Series로 생성. index가 컬럼명에 유의
coeff = pd.Series(data=np.round(lr.coef_, 1), index=X_data.columns)
print(coeff.sort_values(ascending=False))
"""
AveBedrms     0.6
MedInc        0.4
HouseAge      0.0
Population   -0.0
AveOccup     -0.0
AveRooms     -0.1
Latitude     -0.4
Longitude    -0.4
dtype: float64
"""

from sklearn.model_selection import cross_val_score

y_target = caliDF['PRICE']
X_data = caliDF.drop(['PRICE'], axis=1, inplace=False)
lr = LinearRegression()

# cross_val_score()로 5 Fold 셋으로 MSE를 구한 뒤 이를 기반으로 다시 RMSE 구함
neg_mse_scores = cross_val_score(lr, X_data, y_target, scoring="neg_mean_squared_error", cv=5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# cross_val_score(scoring="neg_mean_squared_error")로 반환된 값은 모두 음수
print(' 5 folds 의 개별 Negative MSE scores:', np.round(neg_mse_scores, 2))
print(' 5 folds 의 개별 RMSE scores:', np.round(rmse_scores, 2))
print(f' 5 folds 의 평균 RMSE: {avg_rmse:.3f}')