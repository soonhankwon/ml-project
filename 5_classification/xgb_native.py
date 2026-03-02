import xgboost as xgb

print(xgb.__version__) # 3.1.3

# 파이썬 Native XGBoost 적용 - 위스콘신 Breast Cancer 데이터 셋
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

dataset = load_breast_cancer()
features = dataset.data
labels = dataset.target

cancer_df = pd.DataFrame(data=features, columns=dataset.feature_names)
cancer_df['target'] = labels
print(cancer_df.head(3))
print(dataset.target_names)
print(cancer_df['target'].value_counts())
"""
   mean radius  mean texture  mean perimeter  mean area  ...  worst concave points  worst symmetry  worst fractal dimension  target
0        17.99         10.38           122.8     1001.0  ...                0.2654          0.4601                  0.11890       0
1        20.57         17.77           132.9     1326.0  ...                0.1860          0.2750                  0.08902       0
2        19.69         21.25           130.0     1203.0  ...                0.2430          0.3613                  0.08758       0

[3 rows x 31 columns]

['malignant' 'benign']
target
1    357
0    212
Name: count, dtype: int64
"""

# cancer_df에서 feature용 DataFrame과 Label용 Series 객체 추출
# 맨 마지막 컬럼이 Label이므로 Feature용 DataFrame은 cancer_df의 첫번째 컬럼에서 맨 마지막 두번째 컬럼까지를 :-1 슬라이싱으로 추출
X_feature = cancer_df.iloc[:, :-1]
y_label = cancer_df.iloc[:, -1]

# 전체 데이터 중 80% 학습용 데이터, 20% 테스트용 데이터 추출
X_train, X_test, y_train, y_test = train_test_split(X_feature, y_label, test_size=0.2, random_state=156)
# 위에서 만든 X_train, y_train을 다시 쪼개서 90%는 학습과 10%는 검증용 데이터로 분리
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=156)

print(X_train.shape, X_test.shape)
print(X_tr.shape, X_val.shape)
"""
(455, 30) (114, 30)
(409, 30) (46, 30)
"""

# 학습과 예측 데이터 세트를 DMatrix로 변환, DMatrix는 넘파이 array, DataFrame에서도 변환 가능
# 학습, 검증, 테스트용 DMatrix를 생성
dtr = xgb.DMatrix(data=X_tr, label=y_tr)
dval = xgb.DMatrix(data=X_val, label=y_val)
dtest = xgb.DMatrix(data=X_test, label=y_test)

# 하이퍼 파라미터 설정
params = {
    'max_depth': 3,
    'eta': 0.05,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}
num_rounds = 400

# 주어진 하이퍼 파리미터와 early stopping 파라미터를 train()함수의 파라미터로 전달하고 학습
# 학습 데이터 셋은 train 또는 평가 데이터셋은 eval로 명기
eval_list = [(dtr, 'train'), (dval, 'eval')]

# 하이퍼 파리머터와 early stopping 파라미터를 train() 함수의 파라미터로 전달
xgb_model = xgb.train(params=params, dtrain=dtr, num_boost_round=num_rounds, 
early_stopping_rounds=50, evals=eval_list)

# predict()를 통해 예측 확률값을 반환하고 예측 값으로 변환
pred_probs = xgb_model.predict(dtest)
print('predict() 수행 결과값을 10개만 표시, 예측 확률 값으로 표시됨')
print(np.round(pred_probs[:10], 3))

# 예측 확률이 0.5 보다 크면 1, 그렇지 않으면 0으로 예측값 결정하여 List 객체인 preds에 저장
preds = [1 if x > 0.5 else 0 for x in pred_probs]
print('예측값 10개만 표시:', preds[:10])
print(pred_probs.shape)

"""
predict() 수행 결과값을 10개만 표시, 예측 확률 값으로 표시됨
[0.938 0.004 0.776 0.058 0.975 1.    0.999 0.999 0.998 0.   ]
예측값 10개만 표시: [1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
(114,)
"""

# get_clf_eval()을 통해 예측 평가

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    # ROC-AUC 추가
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    print(f'정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}, AOC: {roc_auc:.4f}')

get_clf_eval(y_test, preds, pred_probs)
"""
오차 행렬
[[35  2]
 [ 2 75]]
정확도: 0.9649, 정밀도: 0.9740, 재현율: 0.9740, F1: 0.9740, AOC: 0.9965
"""

# Feature Importance 시각화
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax)
plt.show()