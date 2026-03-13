import numpy as np
import pandas as pd

cust_df = pd.read_csv("./santander_customer_satisfaction/train.csv", encoding='latin-1')
print('dataset shape:', cust_df.shape)
print(cust_df.head(3))
"""
dataset shape: (76020, 371)
   ID  var3  var15  ...  saldo_medio_var44_ult3     var38  TARGET
0   1     2     23  ...                     0.0  39205.17       0
1   3     2     34  ...                     0.0  49278.03       0
2   4     2     23  ...                     0.0  67333.77       0

[3 rows x 371 columns]
"""

print(cust_df.info())
"""
<class 'pandas.DataFrame'>
RangeIndex: 76020 entries, 0 to 76019
Columns: 371 entries, ID to TARGET
dtypes: float64(111), int64(260)
memory usage: 215.2 MB
None
"""

print(cust_df['TARGET'].value_counts())
unsatisfied_cnt = cust_df[cust_df['TARGET'] == 1].TARGET.count()
total_cnt = cust_df.TARGET.count()
print(f'unsatisfied 비율은 {(unsatisfied_cnt / total_cnt):.2f}')
"""
TARGET
0    73012
1     3008
Name: count, dtype: int64
unsatisfied 비율은 0.04
"""

print(cust_df.describe())
"""
                  ID           var3         var15  ...  saldo_medio_var44_ult3         var38        TARGET
count   76020.000000   76020.000000  76020.000000  ...            76020.000000  7.602000e+04  76020.000000
mean    75964.050723   -1523.199277     33.212865  ...               56.614351  1.172358e+05      0.039569
std     43781.947379   39033.462364     12.956486  ...             2852.579397  1.826646e+05      0.194945
min         1.000000 -999999.000000      5.000000  ...                0.000000  5.163750e+03      0.000000
25%     38104.750000       2.000000     23.000000  ...                0.000000  6.787061e+04      0.000000
50%     76043.000000       2.000000     28.000000  ...                0.000000  1.064092e+05      0.000000
75%    113748.750000       2.000000     40.000000  ...                0.000000  1.187563e+05      0.000000
max    151838.000000     238.000000    105.000000  ...           397884.300000  2.203474e+07      1.000000

[8 rows x 371 columns]
"""

print(cust_df['var3'].value_counts())
"""
var3
 2         74165
 8           138
-999999      116
 9           110
 3           108
           ...  
 63            1
 194           1
 40            1
 57            1
 87            1
Name: count, Length: 208, dtype: int64
"""

# var3 피처값 대체 및 ID 피처 드롭
cust_df['var3'] = cust_df['var3'].replace(-999999, 2)
cust_df.drop('ID', axis=1, inplace=True)

# 피처 세트와 레이블 세트분리. 레이블 컬럼은 DataaFrame의 맨 마지막에 위치해 컬럼 위치 -1로 분리
X_features = cust_df.iloc[:, :-1] # 맨마지막 컬럼 제외
y_labels = cust_df.iloc[:, -1] # 맨마지막 컬럼만
print(f'피처 데이터 shape:{X_features.shape}')
"""
피처 데이터 shape:(76020, 369)
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=0, stratify=y_labels)
train_cnt = y_train.count()
test_cnt = y_test.count()
print(f'학습 세트 Shape:{X_train.shape}, 테스트 세트 Shape:{X_test.shape}')
print('학습 세트 레이블 값 분포 비율')
print(y_train.value_counts() / train_cnt)
print('테스트 세트 레이블 값 분포 비율')
print(y_test.value_counts() / test_cnt)

"""
학습 세트 Shape:(60816, 369), 테스트 세트 Shape:(15204, 369)
학습 세트 레이블 값 분포 비율
TARGET
0    0.960438
1    0.039562
Name: count, dtype: float64
테스트 세트 레이블 값 분포 비율
TARGET
0    0.960405
1    0.039595
Name: count, dtype: float64
"""

# X_train, y_train을 다시 학습과 검증 데이터 세트로 분리
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

# XGBoost 모델 학습과 하이퍼 파라미터 튜닝
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.metrics import roc_auc_score

# n_estimators 500, learning_rate 0.05, 성능 평가 지표를 auc, 조기 중단 파라미터는 100으로 설정하고 학습 수행
xgb_clf = XGBClassifier(
    n_estimators=500, learning_rate=0.05, random_state=156,
    eval_metric='auc',
    callbacks=[EarlyStopping(rounds=100, metric_name='auc', maximize=True)],
)
xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=True)
xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1])
print(f'ROC AUC: {xgb_roc_score:.4f}')
"""
ROC AUC: 0.8020
"""

from hyperopt import hp

# max_depth는 5에서 15까지 1간격, min_child_weight는 1에서 6까지 1간격
# colsample_bytree는 0.5에서 0.95 사이, learning_rate는 0.01에서 0.2사이 정규분포값으로 검색
xgb_search_space = {
    'max_depth': hp.quniform('max_depth', 5, 15, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.95),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2)
}

from sklearn.model_selection import KFold

# 목적함수 설정
# 추후 fmin() 에서 입력된 search_space값으로 XGBClassifier 교차 검증 학습 후 -1 * roc_auc 평균값 반환
def objective_func(search_space):
    # 수행 시간 절약을 위해 n_estimators는 100으로 축소, early stopping은 30회로 설정
    xgb_clf = XGBClassifier(
        n_estimators=100, 
        learning_rate=search_space['learning_rate'], 
        max_depth=int(search_space['max_depth']),
        min_child_weight=int(search_space['min_child_weight']),
        colsample_bytree=search_space['colsample_bytree'],
        eval_metric='auc',
        callbacks=[EarlyStopping(rounds=30, metric_name='auc', maximize=True)])

    # 3개 K-fold 방식으로 평가된 roc_auc 지표를 담는 list
    roc_auc_list = []

    # 3개 K-fold 방식 적용
    kf = KFold(n_splits=3)

    # X_train을 다시 학습과 검증용 데이터로 분리
    for tr_index, val_index in kf.split(X_train):
        # kf.split(X_train)으로 추출된 학습과 검증 index값으로 학습과 검증 데이터 세트 분리
        X_tr, y_tr = X_train.iloc[tr_index], y_train.iloc[tr_index]
        X_val, y_val = X_train.iloc[val_index], y_train.iloc[val_index]
        
        # 추출된 학습과 검증 데이터로 학습 수행
        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=True)

        # 1로 예측한 확률값 추출후 roc auc 계산, 평균 roc auc 계산을 위해 list에 결과값을 담음
        score = roc_auc_score(y_val, xgb_clf.predict_proba(X_val)[:, 1])
        roc_auc_list.append(score)

    # 3개 K-fold로 계산된 roc_auc의 평균값을 반환
    # HyperOpt는 목적함수의 최소값을 위한 입력값을 찾으므로 -1을 곱한뒤 리턴
    return -1 * np.mean(roc_auc_list)

from hyperopt import fmin, tpe, Trials

trials = Trials()

# fmin() 함수를 호출, max_evals 지정된 횟수만큼 반복 후 목적함수의 최소값을 가지는 최적 입력값 추출
best = fmin(
    fn=objective_func,
    space=xgb_search_space,
    algo=tpe.suggest,
    max_evals=50, # 최대 반복 횟수
    trials=trials
)

print('best:', best)
"""
best: {'colsample_bytree': np.float64(0.9446553138681885), 'learning_rate': np.float64(0.03910180288653287), 'max_depth': np.float64(6.0), 'min_child_weight': np.float64(6.0)}
"""

# n_estimator를 500 증가 후 최적으로 찾은 하이퍼 파라미터 기반으로 학습, 예측 수행
xgb_clf = XGBClassifier(
    n_estimators=500, 
    learning_rate=round(best['learning_rate'], 5),
    max_depth=int(best['max_depth']),
    min_child_weight=int(best['min_child_weight']),
    colsample_bytree=round(best['colsample_bytree'], 5),
    eval_metric='auc',
    callbacks=[EarlyStopping(rounds=100, metric_name='auc', maximize=True)],
)

xgb_clf.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr)], verbose=True)
xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1])
print(f'ROC AUC: {xgb_roc_score:.4f}')
"""
ROC AUC: 0.8198
"""

from xgboost import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(xgb_clf, ax=ax, max_num_features=20, height=0.4)
plt.show()

# 캐글 제출용 CSV 생성
test_df = pd.read_csv("./santander-customer-satisfaction/test.csv", encoding='latin-1')
test_ids = test_df['ID'].copy()
test_df['var3'] = test_df['var3'].replace(-999999, 2)
test_df.drop('ID', axis=1, inplace=True)
X_test_sub = test_df
pred_proba = xgb_clf.predict_proba(X_test_sub)[:, 1]
submission = pd.DataFrame({'ID': test_ids, 'TARGET': pred_proba})
submission.to_csv("./santander-customer-satisfaction/submission.csv", index=False)
print(f'제출 파일 생성 완료: submission.csv ({len(submission)} rows)')