from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

housing = fetch_california_housing()
caliDF = pd.DataFrame(housing.data, columns=housing.feature_names)
caliDF['PRICE'] = housing.target

y_target = caliDF['PRICE']
X_data = caliDF.drop(['PRICE'], axis=1, inplace=False)

ridge = Ridge(alpha=10)
neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring="neg_mean_squared_error", cv=5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)
print(' 5 folds의 개별 Negative MSE scores:', np.round(neg_mse_scores, 3))
print(' 5 folds의 개별 RMSE scores:', np.round(rmse_scores, 3))
print(f' 5 folds의 평균 RMSE: {avg_rmse:.3f}')
"""
 5 folds의 개별 Negative MSE scores: [-0.484 -0.623 -0.646 -0.544 -0.494]
 5 folds의 개별 RMSE scores: [0.695 0.789 0.804 0.737 0.703]
 5 folds의 평균 RMSE: 0.746
"""

# alpha값을 0, 0.1, 1, 10, 100으로 변경하면서 RMSE 측정
# Ridge에 사용될 alpha 파라미터의 값들을 정의
alphas = [0, 0.1, 1, 10, 100]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)

    # cross_val_score를 이용하여 5 fold의 평균 RMSE 계산
    neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring="neg_mean_squared_error", cv=5)
    avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
    print(f'alpha {alpha} 일 때 5 folds의 평균 RMSE: {avg_rmse:.6f}')

"""
alpha 0 일 때 5 folds의 평균 RMSE: 0.745907
alpha 0.1 일 때 5 folds의 평균 RMSE: 0.745906
alpha 1 일 때 5 folds의 평균 RMSE: 0.745898
alpha 10 일 때 5 folds의 평균 RMSE: 0.745821
alpha 100 일 때 5 folds의 평균 RMSE: 0.745562
"""

# 각 alpha에 따른 회귀 계수 값을 시각화. 각 alpha값 별로 plt.subplots로 맷플롯립 축 생성
# 각 alpha에 따른 회귀 계수 값을 시각화하기 위해 5개의 열로 된 맷플롯핍 축 생성
fig, axs = plt.subplots(figsize=(18, 6), nrows=1, ncols=5)
# 각 alpha에 따른 회귀 계수 값을 데이터로 저장하기 위한 DataFrame 생성
coeff_df = pd.DataFrame()

# alphas 리스트 값을 차례로 입력해 회귀 계수 값 시각화 및 데이터 저장, pos는 axis의 위치 지정
for pos, alpha in enumerate(alphas):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_data, y_target)
    # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame 컬럼으로 추가
    coeff= pd.Series(data=ridge.coef_, index=X_data.columns)
    colname = 'alpha:' + str(alpha)
    coeff_df[colname] = coeff
    # 막대 그래프로 각 alpha 값에서의 회귀 계수를 시각화. 회귀 계수값이 높은 순으로 표현
    coeff = coeff.sort_values(ascending=False)
    axs[pos].set_title(colname)
    axs[pos].set_xlim(-3, 6)
    sns.barplot(x=coeff.values, y=coeff.index, ax=axs[pos])

# plt.show()

# alpha값에 따른 컬럼별 회귀계수 출력
ridge_alphas = [0 , 0.1 , 1 , 10 , 100]
sort_column = 'alpha:'+str(ridge_alphas[0])
print(coeff_df.sort_values(by=sort_column, ascending=False))
"""
             alpha:0  alpha:0.1   alpha:1  alpha:10  alpha:100
AveBedrms   0.645066   0.644965  0.644062  0.635174   0.558249
MedInc      0.436693   0.436683  0.436594  0.435719   0.428210
HouseAge    0.009436   0.009436  0.009437  0.009452   0.009592
Population -0.000004  -0.000004 -0.000004 -0.000004  -0.000003
AveOccup   -0.003787  -0.003787 -0.003786 -0.003785  -0.003773
AveRooms   -0.107322  -0.107303 -0.107133 -0.105456  -0.091012
Latitude   -0.421314  -0.421313 -0.421299 -0.421156  -0.419061
Longitude  -0.434514  -0.434511 -0.434485 -0.434217  -0.430993
"""