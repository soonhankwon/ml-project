from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, ElasticNet, Ridge
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np

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
        if model_name =='Ridge': model = Ridge(alpha=param)
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

# 라쏘에 사용될 alpha 파라미터의 값들을 정의하고 get_linear_reg_eval() 함수 호출
lasso_alphas = [0.07, 0.1, 0.5, 1, 3]
coeff_lasso_df = get_linear_reg_eval('Lasso', params=lasso_alphas, X_data_n=X_data, y_target_n=y_target)

# 반환된 coeff_lasso_df를 첫번째 컬럼순으로 내림차순 정렬하여 회귀계수 DataFrame 출력
sort_column = 'alpha:' + str(lasso_alphas[0])
print(coeff_lasso_df.sort_values(by=sort_column, ascending=False))
"""
#######  Lasso #######
alpha 0.07일 때 5 폴드 세트의 평균 RMSE: 0.784
alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 0.813
alpha 0.5일 때 5 폴드 세트의 평균 RMSE: 0.873
alpha 1일 때 5 폴드 세트의 평균 RMSE: 1.000
alpha 3일 때 5 폴드 세트의 평균 RMSE: 1.171
            alpha:0.07  alpha:0.1  alpha:0.5   alpha:1   alpha:3
MedInc        0.387057   0.390583   0.288855  0.145469  0.000000
HouseAge      0.013391   0.015082   0.012031  0.005815  0.000000
Population    0.000010   0.000018   0.000012 -0.000006 -0.000023
AveRooms     -0.000000  -0.000000   0.000000  0.000000  0.000000
AveBedrms     0.000000   0.000000  -0.000000  0.000000  0.000000
AveOccup     -0.003409  -0.003323  -0.000000 -0.000000 -0.000000
Longitude    -0.204689  -0.099225  -0.000000 -0.000000  0.000000
Latitude     -0.212806  -0.114214  -0.000000 -0.000000  0.000000
"""

# 엘라스틱넷 회귀
# 엘라스틱넷에 사용될 alpha 파라미터 값들을 정의하고 get_linear_reg_eval() 함수 호출
# l1_ratio는 0.7로 고정
elastic_alphas = [0.07, 0.1, 0.5, 1, 3]
coeff_elastic_df = get_linear_reg_eval('ElasticNet', params=elastic_alphas, X_data_n=X_data, y_target_n=y_target)
# 반환된 coeff_elastic_df를 첫번째 컬럼순으로 내림차순 정렬하여 회귀계수 DataFrame 출력
sort_column = 'alpha:' + str(elastic_alphas[0])
print(coeff_elastic_df.sort_values(by=sort_column, ascending=False))
"""
#######  ElasticNet #######
alpha 0.07일 때 5 폴드 세트의 평균 RMSE: 0.773
alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 0.788
alpha 0.5일 때 5 폴드 세트의 평균 RMSE: 0.855
alpha 1일 때 5 폴드 세트의 평균 RMSE: 0.931
alpha 3일 때 5 폴드 세트의 평균 RMSE: 1.171
            alpha:0.07  alpha:0.1  alpha:0.5   alpha:1   alpha:3
MedInc        0.384500   0.385980   0.318532  0.213455  0.000000
HouseAge      0.012534   0.013697   0.013662  0.009156  0.000000
Population    0.000007   0.000012   0.000018  0.000003 -0.000023
AveRooms      0.000000  -0.000000   0.000000  0.000000  0.000000
AveBedrms     0.000000   0.000000  -0.000000 -0.000000  0.000000
AveOccup     -0.003502  -0.003437  -0.000837 -0.000000 -0.000000
Longitude    -0.259559  -0.185737  -0.000000 -0.000000 -0.000000
Latitude     -0.264115  -0.195109  -0.000000 -0.000000 -0.000000
"""