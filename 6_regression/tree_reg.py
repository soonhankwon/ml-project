from sklearn.datasets import fetch_california_housing
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