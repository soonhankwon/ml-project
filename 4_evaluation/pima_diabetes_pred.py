import pandas as pd

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