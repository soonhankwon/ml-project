import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv('./titanic_ml_dataset/Titanic_train.csv', sep=',')
print(titanic_df.head(3))
"""
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S

- PassengerId: 탑승자 데이터 일련번호
- Survived: 생존여부
- PClass: 티켓의 선실 등급
- Sex: 탑승자 성별
- Name: 탑승자 이름
- Sibsp: 같이 탑승한 형제자매 또는 배우자 인원수
- Parch: 같이 탑승한 부모님 또는 어린이 인원수
- Ticket: 티켓번호
- Fare: 요금
- Cabin: 선실번호
- Embarked: 중간 정착 항구 (Cherbourg, Queenstown, Southampton)
"""

print('### train 데이터 정보 ###')
print(titanic_df.info())
"""
### train 데이터 정보 ###
<class 'pandas.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    str    
 4   Sex          891 non-null    str    
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    str    
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    str    
 11  Embarked     889 non-null    str    
dtypes: float64(2), int64(5), str(5)
memory usage: 83.7 KB
None
"""

print(titanic_df.describe())
"""
       PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
"""

# NULL 컬럼들에 대한 처리
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('N')
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('N')
print('데이터 세트 Null 값 갯수: ', titanic_df.isnull().sum().sum())
# 데이터 세트 Null 값 갯수:  0

# 주요 컬럼 EDA
print(titanic_df.dtypes[titanic_df.dtypes == 'str'].index.tolist())
print('\nSex 값 분포: \n', titanic_df['Sex'].value_counts())
print('\nCabin 값 분포: \n', titanic_df['Cabin'].value_counts())
print('\nEmbarked 값 분포: \n', titanic_df['Embarked'].value_counts())
"""
['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

Sex 값 분포: 
 Sex
male      577
female    314
Name: count, dtype: int64

Cabin 값 분포: 
 Cabin
N              687
G6               4
C23 C25 C27      4
B96 B98          4
F33              3
              ... 
E17              1
A24              1
C50              1
B42              1
C148             1
Name: count, Length: 148, dtype: int64

Embarked 값 분포: 
 Embarked
S    644
C    168
Q     77
N      2
"""

titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3))
print(titanic_df['Cabin'].value_counts())
"""
0    N
1    C
2    N
Name: Cabin, dtype: str
Cabin
N    687
C     59
B     47
D     33
E     32
A     15
F     13
G      4
T      1
"""

print(titanic_df.groupby(['Sex', 'Survived'])['Survived'].count())
"""
Sex     Survived
female  0            81
        1           233
male    0           468
        1           109
Name: Survived, dtype: int64
"""

# barplot: 평균값
sns.barplot(x='Sex', y='Survived', data=titanic_df)
# plt.show()
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)
# plt.show()

# 입력 age에 따라 구분값을 반환하는 함수 설정, DataFame의 apply lambda식에 사용
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknow'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'

    return cat

# 막대그래프의 크기 figure를 더 크게 설정
plt.figure(figsize=(10,6))

# X축의 값을 순차적으로 표시하기 위한 설정
group_names = ['Unknow', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

# lambda 식에 위에서 생성한 get_category() 함수를 반화값으로 지정
# get_category(X)는 입력값으로 'Age' 컬럼값을 받아서 해당하는 cat 반환
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: get_category(x))
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
titanic_df.drop('Age_cat', axis=1, inplace=True)

# plt.show()
print(titanic_df['Age'])
"""
0      22.000000
1      38.000000
2      26.000000
3      35.000000
4      35.000000
         ...    
886    27.000000
887    19.000000
888    29.699118
889    26.000000
890    32.000000
Name: Age, Length: 891, dtype: float64
"""

from sklearn.preprocessing import LabelEncoder

def encode_features(dataDF):
    features = ['Cabin', 'Sex', 'Embarked']
    le = LabelEncoder()
    for feature in features:
        le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])

    return dataDF

titanic_df = encode_features(titanic_df)
print(titanic_df.head())
"""
   PassengerId  Survived  Pclass  ...     Fare  Cabin  Embarked
0            1         0       3  ...   7.2500      7         3
1            2         1       1  ...  71.2833      2         0
2            3         1       3  ...   7.9250      7         3
3            4         1       1  ...  53.1000      2         3
4            5         0       3  ...   8.0500      7         3

[5 rows x 12 columns]
"""