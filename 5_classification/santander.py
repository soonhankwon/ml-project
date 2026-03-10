import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

cust_df = pd.read_csv("./santander-customer-satisfaction/train.csv", encoding='latin-1')
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
    callbacks=[EarlyStopping(rounds=100, metric_name='auc', maximize=False)],
)
xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=True)
xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1])
print(f'ROC AUC: {xgb_roc_score:.4f}')
"""
ROC AUC: 0.8020
"""