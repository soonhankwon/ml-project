from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import cross_val_score

housing = fetch_california_housing()
caliDF = pd.DataFrame(housing.data, columns=housing.feature_names)
caliDF['PRICE'] = housing.target

y_target = caliDF['PRICE']
X_data = caliDF.drop(['PRICE'], axis=1, inplace=False)

# alpha값에 따른 회귀 모델의 폴드 평균 RMSE를 출력하고 회귀 계수값들을 DataFrame으로 변환
def get_linear_reg_eval(model_name, params=None, X_data_n=None, y_target_n=None, 
    verbose=True, return_coeff=True):
    coeff_df = pd.DataFrame()
    if verbose : print('####### ', model_name, '#######')
    
    for param in params:
        if model_name =='Ridge': model = Ridge(alpha=param, solver='svd')
        elif model_name =='Lasso': model = Lasso(alpha=param)
        elif model_name =='ElasticNet': model = ElasticNet(alpha=param, l1_ratio=0.7)
        neg_mse_scores = cross_val_score(model, X_data_n, y_target_n, scoring="neg_mean_squared_error", cv=5)
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
        print(f'alpha {param}일 때 5 폴드 세트의 평균 RMSE: {avg_rmse:.3f}')
        # cross_val_score는 evaluation metric만 반환하므로 모델을 다시 학습하여 회귀계수 추출

        model.fit(X_data_n, y_target_n)
        if return_coeff:
            # alpha에 따른 피처별 회귀계수를 Series로 변환하고 이를 DataFrame 컬럼으로 추가
            coeff = pd.Series(data=model.coef_, index=X_data_n.columns)
            colname = 'alpha:' + str(param)
            coeff_df[colname] = coeff
    return coeff_df

# methods는 표준 정규 분포 변환(Standard), 최대/최소값 정규화(MinMax), 로그변환(Log) 결정
# p_degree는 다항식 특성을 추가할 때 적용. p_degree는 2이상 부여하지 않음.
def get_scaled_data(method='None', p_degree=None, input_data=None):
    if method == 'Standard':
        scaled_data = StandardScaler().fit_transform(input_data)
    elif method == 'MinMax':
        scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == 'Log':
        # log1p(x)는 x > -1에서만 실수. 경도(Longitude) 등 음수 피처는 NaN이 되므로 컬럼별로 적용
        if isinstance(input_data, pd.DataFrame):
            scaled_data = input_data.copy()
            for col in scaled_data.columns:
                s = scaled_data[col]
                if s.min() > -1:
                    scaled_data[col] = np.log1p(s)
        else:
            arr = np.asarray(input_data, dtype=float)
            scaled_data = arr.copy()
            for j in range(arr.shape[1]):
                col = scaled_data[:, j]
                if np.nanmin(col) > -1:
                    scaled_data[:, j] = np.log1p(col)
    else:
        scaled_data = input_data

    if p_degree != None:
        scaled_data = PolynomialFeatures(degree=p_degree, include_bias=False).fit_transform(scaled_data)

    return scaled_data

# Ridge의 alpha값을 다르게 적용하고 다양한 데이터 변환방법에 따른 RMSE 추출
alphas = [0.1, 1, 10, 100]
# 변환 방법은 모두 6개, 원본 그대로, 표준정규분포, 표준정규분포+다항식 특성
# 최대/최소 정규화, 최대/최소 정규화+다항식 특성, 로그변환
scaled_methods = [(None, None), ('Standard', None), ('Standard', 2), ('MinMax', None), ('MinMax', 2), ('Log', None)]

for scaled_method in scaled_methods:
    X_data_scaled = get_scaled_data(method=scaled_method[0], p_degree=scaled_method[1], input_data=X_data)
    print(X_data_scaled.shape, X_data.shape)
    print(f'\n## 변환유형:{scaled_method[0]}, Polynomial Degree:{scaled_method[1]}')
    get_linear_reg_eval('Ridge', params=alphas, X_data_n=X_data_scaled, y_target_n=y_target, verbose=False, return_coeff=False)

"""
(20640, 8) (20640, 8)

## 변환유형:None, Polynomial Degree:None
alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 0.746
alpha 1일 때 5 폴드 세트의 평균 RMSE: 0.746
alpha 10일 때 5 폴드 세트의 평균 RMSE: 0.746
alpha 100일 때 5 폴드 세트의 평균 RMSE: 0.746
(20640, 8) (20640, 8)

## 변환유형:Standard, Polynomial Degree:None
alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 0.746
alpha 1일 때 5 폴드 세트의 평균 RMSE: 0.746
alpha 10일 때 5 폴드 세트의 평균 RMSE: 0.746
alpha 100일 때 5 폴드 세트의 평균 RMSE: 0.746
(20640, 44) (20640, 8)

## 변환유형:Standard, Polynomial Degree:2
alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 3.323
alpha 1일 때 5 폴드 세트의 평균 RMSE: 2.850
alpha 10일 때 5 폴드 세트의 평균 RMSE: 1.314
alpha 100일 때 5 폴드 세트의 평균 RMSE: 0.719
(20640, 8) (20640, 8)

## 변환유형:MinMax, Polynomial Degree:None
alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 0.745
alpha 1일 때 5 폴드 세트의 평균 RMSE: 0.748
alpha 10일 때 5 폴드 세트의 평균 RMSE: 0.754
alpha 100일 때 5 폴드 세트의 평균 RMSE: 0.816
(20640, 44) (20640, 8)

## 변환유형:MinMax, Polynomial Degree:2
alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 0.726
alpha 1일 때 5 폴드 세트의 평균 RMSE: 0.742
alpha 10일 때 5 폴드 세트의 평균 RMSE: 0.755
alpha 100일 때 5 폴드 세트의 평균 RMSE: 0.789
(20640, 8) (20640, 8)

## 변환유형:Log, Polynomial Degree:None
alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 0.731
alpha 1일 때 5 폴드 세트의 평균 RMSE: 0.732
alpha 10일 때 5 폴드 세트의 평균 RMSE: 0.754
alpha 100일 때 5 폴드 세트의 평균 RMSE: 0.795
"""