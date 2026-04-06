import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

house_df_org = pd.read_csv('house-prices/train.csv')
house_df = house_df_org.copy()
print(house_df.head(3))
print('데이터 세트의 Shape:', house_df.shape)
print('\n전체 feature 들의 type \n', house_df.dtypes.value_counts())
isnull_series = house_df.isnull().sum()
print('\nNull 컬럼과 그 건수:\n', isnull_series[isnull_series > 0].sort_values(ascending=False))

"""
   Id  MSSubClass MSZoning  LotFrontage  ...  YrSold SaleType SaleCondition SalePrice
0   1          60       RL         65.0  ...    2008       WD        Normal    208500
1   2          20       RL         80.0  ...    2007       WD        Normal    181500
2   3          60       RL         68.0  ...    2008       WD        Normal    223500

[3 rows x 81 columns]

전체 feature 들의 type 
 str        43
int64      35
float64     3
Name: count, dtype: int64

Null 컬럼과 그 건수:
 PoolQC          1453
MiscFeature     1406
Alley           1369
Fence           1179
MasVnrType       872
FireplaceQu      690
LotFrontage      259
GarageType        81
GarageYrBlt       81
GarageFinish      81
GarageQual        81
GarageCond        81
BsmtFinType2      38
BsmtExposure      38
BsmtFinType1      37
BsmtCond          37
BsmtQual          37
MasVnrArea         8
Electrical         1
dtype: int64
"""

# SalePrice skew 확인
plt.title('Original Sale Price Histogram')
plt.xticks(rotation=15)
sns.histplot(house_df['SalePrice'], kde=True)
plt.show()

plt.title('Log transformed Sale Price Histogram')
log_sale_price = np.log1p(house_df['SalePrice'])
sns.histplot(log_sale_price, kde=True)
plt.show()

# SalesPrice 로그 변환
original_sale_price = house_df['SalePrice']
house_df['SalePrice'] = np.log1p(house_df['SalePrice'])

# Null이 너무 많은 컬럼들과 불필요한 컬럼 삭제
house_df.drop(['Id','PoolQC' , 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
# 숫자형 Null은 평균으로 대체 (문자열 컬럼은 mean 불가 → numeric_only)
house_df = house_df.fillna(house_df.mean(numeric_only=True))

# Null값이 있는 피처명과 타입을 추출
null_column_count = house_df.isnull().sum()[house_df.isnull().sum() > 0]
print('## Null 피처의 Type:\n', house_df.dtypes[null_column_count.index])

# one-hot encoding: get_dummies() -> null값을 반영하여 자동 원-핫 인코딩 수행
print('get_dummies() 수행 전 데이터 Shape:', house_df.shape)
house_df_ohe = pd.get_dummies(house_df)
print('get_dummies() 수행 후 데이터 Shape:', house_df_ohe.shape)

null_column_count = house_df_ohe.isnull().sum()[house_df_ohe.isnull().sum() > 0]
print('## Null 피처의 Type:\n', house_df_ohe.dtypes[null_column_count.index])

"""
## Null 피처의 Type:
 MasVnrType      str
BsmtQual        str
BsmtCond        str
BsmtExposure    str
BsmtFinType1    str
BsmtFinType2    str
Electrical      str
GarageType      str
GarageFinish    str
GarageQual      str
GarageCond      str
dtype: object

get_dummies() 수행 전 데이터 Shape: (1460, 75)
get_dummies() 수행 후 데이터 Shape: (1460, 270)

## Null 피처의 Type:
 Series([], dtype: object)
"""

# 선형 회귀 모델의 학습/예측/평가
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice', axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

def get_rmse(model):
   pred = model.predict(X_test)
   mse = mean_squared_error(y_test, pred)
   rmse = np.sqrt(mse)
   print(f'{model.__class__.__name__} 로그 변환된 RMSE: {np.round(rmse, 3)}')

def get_rmses(models):
   rmses = []
   for model in models:
      rmse = get_rmse(model)
      rmses.append(rmse)
   return rmses

# LinearRegression, Ridge, Lasso 학습, 예측, 평가
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

"""
LinearRegression 로그 변환된 RMSE: 0.132
Ridge 로그 변환된 RMSE: 0.127
Lasso 로그 변환된 RMSE: 0.176
"""

def get_top_bottom_coef(model):
   # coef_ 속성을 기반으로 Series 객체를 생성. index는 컬럼명
   coef = pd.Series(model.coef_, index=X_features.columns)

   # + 상위 10개, - 하위 10개 coefficient 추출 & 리턴
   coef_high = coef.sort_values(ascending=False).head(10)
   coef_low = coef.sort_values(ascending=False).tail(10)
   return coef_high, coef_low

def visualize_coefficient(models):
   # 3개 회귀모델의 시각화를 위해 3개의 컬럼을 가지는 subplot 생성
   fig, axs = plt.subplots(figsize=(24, 10), nrows=1, ncols=3)
   fig.tight_layout()
   # 입력인자로 받은 list 객체인 models에서 차례로 model을 추출하여 회귀 계수 시각화
   for i_num, model in enumerate(models):
      # 상위 10개, 하위 10개 회귀 계수를 구하고, 이를 판다스 concat으로 결합
      coef_high, coef_low = get_top_bottom_coef(model)
      coef_concat = pd.concat([coef_high, coef_low])
      # 순차적으로 ax subplot에 barchar로 표현. 한 화면에 표현하기 위해 tick label 위치와 font 크기 조정
      axs[i_num].set_title(model.__class__.__name__+' Coeffiecents', size=25)
      axs[i_num].tick_params(axis='y', direction='in', pad=-120)
      for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):
         label.set_fontsize(22)
         sns.barplot(x=coef_concat.values, y=coef_concat.index , ax=axs[i_num])

# 앞 예제에서 학습한 lr_reg, ridge_reg, lasso_reg 모델 회귀 계수 시각화
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)
plt.show()