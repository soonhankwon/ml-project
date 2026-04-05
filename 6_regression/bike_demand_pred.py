import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 클렌징 및 가공
bike_df = pd.read_csv('bike-sharing-demand/train.csv')
print(bike_df.shape)
print(bike_df.head(3))
"""
(10886, 12)
              datetime  season  holiday  workingday  weather  temp   atemp  humidity  windspeed  casual  registered  count
0  2011-01-01 00:00:00       1        0           0        1  9.84  14.395        81        0.0       3          13     16
1  2011-01-01 01:00:00       1        0           0        1  9.02  13.635        80        0.0       8          32     40
2  2011-01-01 02:00:00       1        0           0        1  9.02  13.635        80        0.0       5          27     32
"""

print(bike_df.info())
"""
<class 'pandas.DataFrame'>
RangeIndex: 10886 entries, 0 to 10885
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   datetime    10886 non-null  str    
 1   season      10886 non-null  int64  
 2   holiday     10886 non-null  int64  
 3   workingday  10886 non-null  int64  
 4   weather     10886 non-null  int64  
 5   temp        10886 non-null  float64
 6   atemp       10886 non-null  float64
 7   humidity    10886 non-null  int64  
 8   windspeed   10886 non-null  float64
 9   casual      10886 non-null  int64  
 10  registered  10886 non-null  int64  
 11  count       10886 non-null  int64  
dtypes: float64(3), int64(8), str(1)
memory usage: 1020.7 KB
None
"""

# 문자열을 datetime 타입으로 변경
bike_df['datetime'] = bike_df.datetime.apply(pd.to_datetime)

# datetime 타입에서 년, 월, 일, 시간 추출
bike_df['year'] = bike_df.datetime.apply(lambda x: x.year)
bike_df['month'] = bike_df.datetime.apply(lambda x: x.month)
bike_df['day'] = bike_df.datetime.apply(lambda x: x.day)
bike_df['hour'] = bike_df.datetime.apply(lambda x: x.hour)

drop_columns = ['datetime', 'casual', 'registered']
bike_df.drop(drop_columns, axis=1, inplace=True)

fig, axs = plt.subplots(figsize=(16, 8), ncols=4, nrows=2)
cat_features = ['year', 'month', 'season', 'weather', 'day', 'hour', 'holiday', 'workingday']
# cat_features에 있는 모든 컬럼별로 개별 컬럼값에 따른 count의 합을 barplot으로 시각화
for i, feature in enumerate(cat_features):
    row = int(i/4)
    col = i%4
    # 사본의 barplot을 이용해 컬럼값에 따른 count의 합을 표현
    sns.barplot(x=feature, y='count', data=bike_df, ax=axs[row][col])

# plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# log 값 변환시 NaN등의 이슈로 log()가 아닌 log1p()를 이용하여 RMSLE 계산
def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

# 사이킷런의 mean_square_error()를 이용하여 RMSE 계산
def rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))

# MSE, RMSE, RMSLE를 모두 계산
def evaluate_regr(y, pred):
    rmsle_val = rmsle(y, pred)
    rmse_val = rmse(y, pred)
    # MAE는 사이킷런의 mean_absolute_error()로 계산
    mae_val = mean_absolute_error(y, pred)
    print(f'RMSLE: {rmsle_val:.3f}, RMSE: {rmse_val:.3f}, MAE: {mae_val:.3f}')

# 로그 변환, 피처 인코딩, 모델 학습/예측/평가
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso

y_target = bike_df['count']
X_features = bike_df.drop(['count'], axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=0)

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

evaluate_regr(y_test, pred)
"""
RMSLE: 1.165, RMSE: 140.900, MAE: 105.924
"""

def get_top_error_data(y_test, pred, n_tops=5):
    # DataFrame에 컬럼들로 실제 대여횟수(count)와 예측값을 서로 비교할 수 있도록 생성
    result_df = pd.DataFrame(y_test.values, columns=['real_count'])
    result_df['predicted_count'] = np.round(pred)
    result_df['diff'] = np.abs(result_df['real_count'] - result_df['predicted_count'])
    # 예측값과 실제값이 가장 큰 데이터 순으로 출력
    print(result_df.sort_values('diff', ascending=False)[:n_tops])

get_top_error_data(y_test, pred, n_tops=5)
"""
      real_count  predicted_count   diff
1618         890            322.0  568.0
3151         798            241.0  557.0
966          884            327.0  557.0
412          745            194.0  551.0
2817         856            310.0  546.0
"""

# y_target.hist() # 타겟값이 skew 되어있음
# y_log_transform = np.log1p(y_target)
# y_log_transform.hist()

# 타겟 컬럼인 count값을 log1p로 로그 변환
y_target_log = np.log1p(y_target)

# 로그 변환된 y_target_log를 반영하여 학습/테스트 데이터 셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target_log, test_size=0.3, random_state=0)
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

# 테스트 데이터셋의 Target값은 Log변환되었으므로 다시 expm1을 이용하여 원래 스케일로 변환
y_test_exp = np.expm1(y_test)

# 예측값 역시 Log 변환된 타겟기반으로 학습되어 예측되었으므로 다시 expm1으로 스케일 변환
pred_exp = np.expm1(pred)

evaluate_regr(y_test_exp, pred_exp)
"""
RMSLE: 1.017, RMSE: 162.594, MAE: 109.286
"""

coef = pd.Series(lr_reg.coef_, index=X_features.columns)
coef_sort = coef.sort_values(ascending=False)
sns.barplot(x=coef_sort.values, y=coef_sort.index)
plt.savefig('log_transform.tif', format='tif', dpi=300, bbox_inches='tight')

# year, month, dat, hour 등의 피처들을 one hot encoding
X_features_ohe = pd.get_dummies(X_features, columns=['year', 'month', 'day', 'hour', 'holiday', 
'workingday', 'season', 'weather'])

# 원-핫 인코딩이 적용된 feature 데이터 세트 기반으로 학습/예측 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size=0.3, random_state=0)

# 모델과 학습/테스트 데이터셋을 입력하면 성능 평가 수치를 반환
def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=False):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if is_expm1:
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)

    print('###', model.__class__.__name__, '###')
    evaluate_regr(y_test, pred)

# model 별로 평가 수행
lr_reg = LinearRegression()
ridge_reg = Ridge(alpha=10)
lasso_reg = Lasso(alpha=0.01)

for model in [lr_reg, ridge_reg, lasso_reg]:
    get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=True)

"""
### LinearRegression ###
RMSLE: 0.590, RMSE: 97.688, MAE: 63.382
### Ridge ###
RMSLE: 0.590, RMSE: 98.529, MAE: 63.893
### Lasso ###
RMSLE: 0.635, RMSE: 113.219, MAE: 72.803
"""

coef = pd.Series(lr_reg.coef_, index=X_features_ohe.columns)
coef_sort = coef.sort_values(ascending=False)[:20]
sns.barplot(x=coef_sort.values, y=coef_sort.index)
# plt.show()

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 랜덤 포레스트, GBM, XGBoost, LightGBM model 별로 평가 수행
rf_reg = RandomForestRegressor(n_estimators=500)
gbm_reg = GradientBoostingRegressor(n_estimators=500)
xgb_reg = XGBRegressor(n_estimators=500)
lgbm_reg = LGBMRegressor(n_estimators=500)

for model in [rf_reg, gbm_reg, xgb_reg, lgbm_reg]:
    get_model_predict(model,X_train.values, X_test.values, y_train.values, y_test.values,is_expm1=True)

"""
### RandomForestRegressor ###
RMSLE: 0.355, RMSE: 50.455, MAE: 31.232
### GradientBoostingRegressor ###
RMSLE: 0.330, RMSE: 53.349, MAE: 32.747
### XGBRegressor ###
RMSLE: 0.339, RMSE: 51.475, MAE: 31.357
### LGBMRegressor ###
RMSLE: 0.319, RMSE: 47.215, MAE: 29.029
"""

# --- sampleSubmission.csv 형식으로 LGBM 제출 파일 생성 (전체 train 학습) ---
OHE_COLUMNS = ['year', 'month', 'day', 'hour', 'holiday', 'workingday', 'season', 'weather']


def _preprocess_bike_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['datetime'] = pd.to_datetime(out['datetime'])
    out['year'] = out['datetime'].dt.year
    out['month'] = out['datetime'].dt.month
    out['day'] = out['datetime'].dt.day
    out['hour'] = out['datetime'].dt.hour
    out = out.drop(columns=['datetime'])
    for col in ('casual', 'registered'):
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def _build_submission_lgbm(
    train_path: str = 'bike-sharing-demand/train.csv',
    test_path: str = 'bike-sharing-demand/test.csv',
    sample_path: str = 'bike-sharing-demand/sampleSubmission.csv',
    out_path: str = 'bike-sharing-demand/submission_lgbm.csv',
) -> None:
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_path)

    X_train_df = _preprocess_bike_features(train_raw)
    y_train_log = np.log1p(X_train_df['count'])
    X_train_df = X_train_df.drop(columns=['count'])

    X_test_df = _preprocess_bike_features(test_raw)

    X_train_ohe = pd.get_dummies(X_train_df, columns=OHE_COLUMNS)
    X_test_ohe = pd.get_dummies(X_test_df, columns=OHE_COLUMNS)
    X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

    lgbm_final = LGBMRegressor(n_estimators=500, random_state=0)
    lgbm_final.fit(X_train_ohe, y_train_log)
    pred_count = np.expm1(lgbm_final.predict(X_test_ohe))
    pred_count = np.clip(pred_count, 0, None)

    out = sample_sub.copy()
    out['count'] = np.round(pred_count).astype(int)
    out.to_csv(out_path, index=False)
    print(f'제출 파일 저장: {out_path} (행 수: {len(out)})')


_build_submission_lgbm()