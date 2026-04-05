from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

housing = fetch_california_housing()
caliDF = pd.DataFrame(housing.data, columns=housing.feature_names)
caliDF['PRICE'] = housing.target

y_target = caliDF['PRICE']
X_data = caliDF.drop(['PRICE'], axis=1, inplace=False)

rf = RandomForestRegressor(random_state=0, n_estimators=100)
neg_mse_scores = cross_val_score(rf, X_data, y_target, scoring="neg_mean_squared_error", cv=5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print(f' 5 교차 검증의 개별 Negative MSE scores: {np.round(neg_mse_scores, 2)}')
print(f' 5 교차 검증의 개별 RMSE scores : {np.round(rmse_scores, 2)}')
print(f' 5 교차 검증의 평균 RMSE : {avg_rmse:.3f} ')
"""
 5 교차 검증의 개별 Negative MSE scores: [-0.5  -0.35 -0.37 -0.44 -0.47]
 5 교차 검증의 개별 RMSE scores : [0.71 0.59 0.61 0.66 0.68]
 5 교차 검증의 평균 RMSE : 0.650 
"""

def get_model_cv_prediction(model, X_data, y_target):
    neg_mse_scores = cross_val_score(model, X_data, y_target, scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-1 * neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    print('#### ', model.__class__.__name__, ' ####')
    print(f' 5 교차 검증의 평균 RMSE: {avg_rmse:.3f}')

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

dt_reg = DecisionTreeRegressor(random_state=0, max_depth=4)
rf_reg = RandomForestRegressor(random_state=0, n_estimators=100)
gb_reg = GradientBoostingRegressor(random_state=0, n_estimators=100)
xgb_reg = XGBRegressor(n_estimators=100)
lgb_reg = LGBMRegressor(n_estimators=100)

models = [dt_reg, rf_reg, gb_reg, xgb_reg, lgb_reg]
# for model in models:
#     get_model_cv_prediction(model, X_data, y_target)

"""
####  DecisionTreeRegressor  ####
 5 교차 검증의 평균 RMSE: 0.809
####  RandomForestRegressor  ####
 5 교차 검증의 평균 RMSE: 0.650
####  GradientBoostingRegressor  ####
 5 교차 검증의 평균 RMSE: 0.642
####  XGBRegressor  ####
 5 교차 검증의 평균 RMSE: 0.660
####  LGBMRegressor  ####
 5 교차 검증의 평균 RMSE: 0.615
"""

import seaborn as sns
import matplotlib.pyplot as plt

rf_reg = RandomForestRegressor(n_estimators=100)

rf_reg.fit(X_data, y_target)

feature_series = pd.Series(data=rf_reg.feature_importances_, index=X_data.columns)
feature_series = feature_series.sort_values(ascending=False)
sns.barplot(x=feature_series, y=feature_series.index)
# plt.show()

"""
MedInc
"""

caliDF_sample = caliDF[['MedInc','PRICE']]
caliDF_sample = caliDF_sample.sample(n=100, random_state=0)
print(caliDF_sample.shape)
"""
(100, 2)
"""
plt.figure()
plt.scatter(caliDF_sample.MedInc, caliDF_sample.PRICE, c="darkorange")
# plt.show()

# 선형회귀와 결정트리 기반의 Regreesor 생성. DecisionTreeRegressor의 max_depth는 각각 2,7
lr_reg = LinearRegression()
rf_reg2 = DecisionTreeRegressor(max_depth=2)
rf_reg7 = DecisionTreeRegressor(max_depth=7)

# 실제 예측을 적용할 테스트용 데이터셋을 4.5 ~ 8.5까지 100개 데이터셋 생성
X_test = np.arange(2.5, 8.5, 0.04).reshape(-1, 1)

# 캘리 주택가격 데이터에서 시각화를 위해 피처는 MedInc만, 결정데이터인 PRICE 추출
X_feature = caliDF_sample['MedInc'].values.reshape(-1, 1)
y_target = caliDF_sample['PRICE'].values.reshape(-1, 1)

# 학습과 예측 수행
lr_reg.fit(X_feature, y_target)
rf_reg2.fit(X_feature, y_target)
rf_reg7.fit(X_feature, y_target)

pred_lr = lr_reg.predict(X_test)
pred_rf2 = rf_reg2.predict(X_test)
pred_rf7 = rf_reg7.predict(X_test)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

# X축값을 4.5 ~ 8.5로 변환하여 입력했을때, 선형회귀와 결정트리 회귀 예측선 시각화
# 선형회귀로 학습된 모델 회귀 예측선
ax1.set_title('Linear Regression')
ax1.scatter(caliDF_sample.MedInc, caliDF_sample.PRICE, c="darkorange")
ax1.plot(X_test, pred_lr, label='linear', linewidth=2)

# DecisionTreeRegressor의 max_depth를 2로 했을 때 회귀 예측선 
ax2.set_title('Decision Tree Regression: \n max_depth=2')
ax2.scatter(caliDF_sample.MedInc, caliDF_sample.PRICE, c="darkorange")
ax2.plot(X_test, pred_rf2, label="max_depth:3", linewidth=2 )

# DecisionTreeRegressor의 max_depth를 7로 했을 때 회귀 예측선 
ax3.set_title('Decision Tree Regression: \n max_depth=7')
ax3.scatter(caliDF_sample.MedInc, caliDF_sample.PRICE, c="darkorange")
ax3.plot(X_test, pred_rf7, label="max_depth:7", linewidth=2)

plt.show()
