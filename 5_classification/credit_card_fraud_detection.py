import pandas as pd
import numpy as np

card_df = pd.read_csv("./creditcard_2013/creditcard.csv")
print(card_df.head(3))
print(card_df.shape)
"""
   Time        V1        V2        V3        V4        V5  ...       V25       V26       V27       V28  Amount  Class
0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  ...  0.128539 -0.189115  0.133558 -0.021053  149.62      0
1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018  ...  0.167170  0.125895 -0.008983  0.014724    2.69      0
2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  ... -0.327642 -0.139097 -0.055353 -0.059752  378.66      0

[3 rows x 31 columns]
(284807, 31)
"""

# 원본 DataFrame은 유지하고 데이터 가공을 위한 DataFrame을 복사하여 반환
# 인자로 입력받은 DataFrame을 복사한 뒤 Time 컬럼만 삭제하고 복사된 DataFrame을 반환
# def get_preprocessed_df(df=None):
#     df_copy = df.copy()
#     df_copy.drop('Time', axis=1, inplace=True)
#     return df_copy

from sklearn.preprocessing import StandardScaler
# 사이킷런의 StandardScaler를 이용하여 정규분포 형태로 Amount 피처값 변환하는 로직으로 수정
# def get_preprocessed_df(df=None):
#     df_copy = df.copy()
#     scaler = StandardScaler()
#     amount_n = scaler.fit_transform(df_copy['Amount'].values.reshape(-1, 1))
#     # 변환된 Amount를 Amount_Scaled로 피처명 변경후 DataFrame 맨 앞 컬럼으로 입력
#     df_copy.insert(0, 'Amount_Scaled', amount_n)
#     # 기존 Time, Amount 피처 삭제
#     df_copy.drop(['Time', 'Amount'], axis=1, inplace=True)
#     return df_copy

def get_outlier(df=None, column=None, weight=1.5):
    # fraud에 해당하는 column 데이터만 추출, 1/4 분위와 3/4 분위 지점을 np.percentile로 구함. 
    fraud = df[df['Class']==1][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
    # IQR을 구하고, IQR에 1.5를 곱하여 최대값과 최소값 지점 구함. 
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight
    # 최대값 보다 크거나, 최소값 보다 작은 값을 아웃라이어로 설정하고 DataFrame index 반환. 
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index
    return outlier_index

# get_processed_df()를 로그 변환 후 V14 피처의 이상치 데이터를 삭제하는 로직으로 변경
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time','Amount'], axis=1, inplace=True)
    # 이상치 데이터 삭제하는 로직 추가
    outlier_index = get_outlier(df=df_copy, column='V14', weight=1.5)
    df_copy.drop(outlier_index, axis=0, inplace=True)
    return df_copy

from sklearn.model_selection import train_test_split
# 학습과 테스트 데이터 세트를 반환하는 함수 생성, 사전 데이터 처리가 끝난 뒤 해당 함수 호출
# 사전 데이터 가공 후 학습과 테스트 데이터 세트를 반환하는 함수
def get_train_test_dataset(df=None):
    # 인자로 입력된 DataFrame의 사전 데이터 가공이 완료된 복사 DataFrame 반환
    df_copy = get_preprocessed_df(df)
    # DataFrame의 맨 마지막 컬럼이 레이블, 나머지는 피처들
    X_features = df_copy.iloc[:, :-1]
    y_target = df_copy.iloc[:, -1]
    # train_test_split()으로 학습과 테스트 데이터 분할, stratify=y_target으로 Stratified 기반 분할
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, stratify=y_target)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

print('학습 데이터 레이블 값 비율')
print(y_train.value_counts() / y_train.shape[0] * 100)
print('테스트 데이터 레이블 값 비율')
print(y_test.value_counts() / y_test.shape[0] * 100)
"""
학습 데이터 레이블 값 비율
Class
0    99.827451
1     0.172549
Name: count, dtype: float64
테스트 데이터 레이블 값 비율
Class
0    99.826785
1     0.173215
Name: count, dtype: float64
"""

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print(f'정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}')

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=1000)
# lr_clf.fit(X_train, y_train)
# lr_pred = lr_clf.predict(X_test)
# lr_pred_proba = lr_clf.predict_proba(X_test)[:, 1]

# get_clf_eval(y_test, lr_pred, lr_pred_proba)
"""
오차 행렬
[[85281    14]
 [   63    85]]
정확도: 0.9992, 정밀도: 0.8348, 재현율: 0.6486, F1: 0.7300, AUC: 0.9648
언밸런스 데이터셋이기 때문에 정확도는 의미가 없다. 
재현율 0.6486 을 높히는데 중점을 둬야함 (TP / (TP+FN) 사기패턴 탐지)
"""

# 인자로 사이킷런의 Estimator 객체와, 학습/테스트 데이터 세트를 입력 받아서 학습/예측/평가 수행
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:, 1]
    get_clf_eval(tgt_test, pred, pred_proba)

from lightgbm import LGBMClassifier

# LightGBM 학습/예측/평가
# boost_from_average: True가 Default 레이블 값이 극도로 불균형한 경우 False가 유리
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=1, boost_from_average=False)
# get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)
"""
오차 행렬
[[85288     7]
 [   24   124]]
정확도: 0.9996, 정밀도: 0.9466, 재현율: 0.8378, F1: 0.8889, AUC: 0.9763
"""

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Amount (원본)')
plt.xticks(range(0, 30000, 1000), rotation=60)
sns.histplot(card_df['Amount'], bins=100, kde=True)

plt.subplot(1, 2, 2)
plt.title('Amount_Scaled (log1p 변환)')
card_df_preprocessed = get_preprocessed_df(card_df)
sns.histplot(card_df_preprocessed['Amount_Scaled'], bins=100, kde=True)
plt.tight_layout()
# plt.show()

# Amount를 정규분포 형태로 변환 후 회귀 및 LightGBM 수행
X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

plt.figure(figsize=(9, 9))
card_df_preprocessed = get_preprocessed_df(card_df)
corr = card_df_preprocessed.corr()
sns.heatmap(corr, cmap='RdBu')
# plt.show()

outlier_index = get_outlier(df=card_df, column='V14', weight=1.5)
print('이상치 데이터 인덱스:', outlier_index)
"""
이상치 데이터 인덱스: Index([8296, 8615, 9035, 9252], dtype='int64')
"""

print('### 로지스틱 회귀 예측 성능 ###')
lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

print('### LightGBM 예측 성능 ###')
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)