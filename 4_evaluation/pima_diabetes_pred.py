import pandas as pd
import numpy as np

diabetes_data = pd.read_csv('./diabetes.csv')
print(diabetes_data['Outcome'].value_counts())
print(diabetes_data.head(3))

"""
Outcome
0    500
1    268
Name: count, dtype: int64
   Pregnancies  Glucose  BloodPressure  SkinThickness  ...   BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35  ...  33.6                     0.627   50        1
1            1       85             66             29  ...  26.6                     0.351   31        0
2            8      183             64              0  ...  23.3                     0.672   32        1

[3 rows x 9 columns]
"""
"""
Pregnancies: 임신횟수
Glucose: 포도당 부하 검사 수치
BloodPressure: 혈압(mm Hg)
SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
Insulin: 혈청 인슐린(mu U/ml)
BMI: 체질량지수
DiabetesPredigreeFucntion: 당뇨 내력 가중치 값
Age: 나이
Outcome: 클래스 결정 값(0또는1)
"""

print(diabetes_data.info())
"""
<class 'pandas.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
None
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 수정된 get_clf_eval() 함수
def get_clf_eval(y_test, pred=None, pred_proba=None):
   confusion = confusion_matrix(y_test, pred)
   accuracy = accuracy_score(y_test, pred)
   precision = precision_score(y_test, pred)
   recall = recall_score(y_test, pred)
   f1 = f1_score(y_test, pred)
   roc_auc = roc_auc_score(y_test, pred)
   print('오차행렬')
   print(confusion)
   print(f'정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}')

def precision_recall_curve_plot(y_test=None, pred_proba_c1=None):
   # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출
   precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

   # X출을 threshold값으로 Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
   plt.figure(figsize=(8,6))
   threshold_boundary = thresholds.shape[0]
   plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
   plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')

   # threshold 값 X축의 Scale을 0.1 단위로 변경
   start, end = plt.xlim()
   plt.xticks(np.round(np.arange(start, end, 0.1), 2))

   # x축, y축 label과 legend, grid 설정
   plt.xlabel('Threshold value')
   plt.ylabel('Precision and Recall value')
   plt.legend()
   plt.grid()
   plt.show()

# 피처 데이터 세트 X, 레이블 데이터 세트 y를 추출
# 맨 끝이 Outcome 컬럼으로 레이블 값. 컬럼 위치 -1을 이용해 추출
X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

# 로지스틱 회귀로 학습, 예측 및 평가 수행
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

get_clf_eval(y_test, pred, pred_proba)

"""
오차행렬
[[87 13]
 [22 32]]
정확도: 0.7727, 정밀도: 0.7111, 재현율: 0.5926, F1: 0.6465, AUC: 0.7313
"""
pred_proba_c1 = lr_clf.predict_proba(X_test)[:, 1]
# precision_recall_curve_plot(y_test, pred_proba_c1)

print(diabetes_data.describe())
"""
       Pregnancies     Glucose  BloodPressure  SkinThickness  ...         BMI  DiabetesPedigreeFunction         Age     Outcome
count   768.000000  768.000000     768.000000     768.000000  ...  768.000000                768.000000  768.000000  768.000000
mean      3.845052  120.894531      69.105469      20.536458  ...   31.992578                  0.471876   33.240885    0.348958
std       3.369578   31.972618      19.355807      15.952218  ...    7.884160                  0.331329   11.760232    0.476951
min       0.000000    0.000000       0.000000       0.000000  ...    0.000000                  0.078000   21.000000    0.000000
25%       1.000000   99.000000      62.000000       0.000000  ...   27.300000                  0.243750   24.000000    0.000000
50%       3.000000  117.000000      72.000000      23.000000  ...   32.000000                  0.372500   29.000000    0.000000
75%       6.000000  140.250000      80.000000      32.000000  ...   36.600000                  0.626250   41.000000    1.000000
max      17.000000  199.000000     122.000000      99.000000  ...   67.100000                  2.420000   81.000000    1.000000

[8 rows x 9 columns]
"""

plt.hist(diabetes_data['Glucose'], bins=100)
# plt.show()

# 0값을 검사할 피처명 리스트 객체 설정
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 전체 데이터 건수
total_count = diabetes_data['Glucose'].count()

# 피처별로 반복하면서 데이터 값이 0인 데이터 건수 추출, 퍼센트 계산
for feature in zero_features:
   zero_count = diabetes_data[diabetes_data[feature] == 0][feature].count()
   print(f'{feature} 0건수는 {zero_count}, 퍼센트는 {100*zero_count/total_count:.2f}')
"""
Glucose 0건수는 5, 퍼센트는 0.65
BloodPressure 0건수는 35, 퍼센트는 4.56
SkinThickness 0건수는 227, 퍼센트는 29.56
Insulin 0건수는 374, 퍼센트는 48.70
BMI 0건수는 11, 퍼센트는 1.43
"""

# zero_features 리스트 내부에 저장된 개별 피처들에 대해서 0값을 평균 값으로 대체
mean_zero_features = diabetes_data[zero_features].mean()
diabetes_data[zero_features] = diabetes_data[zero_features].replace(0, mean_zero_features)

X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]

# StandardScaler 클래스를 이용해 피처 데이터 세트에 일괄적으로 스케일링 적용
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 156, stratify=y)

# 로지스틱 회귀로 학습, 예측 및 평가 수행
lr_clf = LogisticRegression()
lr_clf.fit(X_train , y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

get_clf_eval(y_test , pred, pred_proba)
"""
오차행렬
[[90 10]
 [21 33]]
정확도: 0.7987, 정밀도: 0.7674, 재현율: 0.6111, F1: 0.6804, AUC: 0.7556
"""

from sklearn.preprocessing import Binarizer

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
   # threwsholds 리스트 객체내의 값을 차례로 반복하면서 Evaluation 수행
   for custom_threshold in thresholds:
      binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
      custom_predict = binarizer.transform(pred_proba_c1)
      print('임곗값:', custom_threshold)
      get_clf_eval(y_test, custom_predict, pred_proba_c1)

thresholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.50]
pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)

"""
오차행렬
[[90 10]
 [21 33]]
정확도: 0.7987, 정밀도: 0.7674, 재현율: 0.6111, F1: 0.6804, AUC: 0.7556
임곗값: 0.3
오차행렬
[[67 33]
 [11 43]]
정확도: 0.7143, 정밀도: 0.5658, 재현율: 0.7963, F1: 0.6615, AUC: 0.7331
임곗값: 0.33
오차행렬
[[72 28]
 [12 42]]
정확도: 0.7403, 정밀도: 0.6000, 재현율: 0.7778, F1: 0.6774, AUC: 0.7489
임곗값: 0.36
오차행렬
[[76 24]
 [15 39]]
정확도: 0.7468, 정밀도: 0.6190, 재현율: 0.7222, F1: 0.6667, AUC: 0.7411
임곗값: 0.39
오차행렬
[[78 22]
 [16 38]]
정확도: 0.7532, 정밀도: 0.6333, 재현율: 0.7037, F1: 0.6667, AUC: 0.7419
임곗값: 0.42
오차행렬
[[84 16]
 [18 36]]
정확도: 0.7792, 정밀도: 0.6923, 재현율: 0.6667, F1: 0.6792, AUC: 0.7533
임곗값: 0.45
오차행렬
[[85 15]
 [18 36]]
정확도: 0.7857, 정밀도: 0.7059, 재현율: 0.6667, F1: 0.6857, AUC: 0.7583
임곗값: 0.48
오차행렬
[[88 12]
 [19 35]]
정확도: 0.7987, 정밀도: 0.7447, 재현율: 0.6481, F1: 0.6931, AUC: 0.7641
임곗값: 0.5
오차행렬
[[90 10]
 [21 33]]
정확도: 0.7987, 정밀도: 0.7674, 재현율: 0.6111, F1: 0.6804, AUC: 0.7556
"""

# 임곗값을 0.48로 설정한 Binarizer 생성
binarizer = Binarizer(threshold=0.48)

# 위에서 구한 lr_clf의 predic_proba() 예측 확률 array에서 1에 해당하는 컬럼값을 Binarizer 변환
pred_th_048 = binarizer.fit_transform(pred_proba[:, 1].reshape(-1,1))
get_clf_eval(y_test, pred_th_048, pred_proba[:, 1])

"""
오차행렬
[[88 12]
 [19 35]]
정확도: 0.7987, 정밀도: 0.7447, 재현율: 0.6481, F1: 0.6931, AUC: 0.7641
"""