import pandas as pd

# read_csv()
titanic_df = pd.read_csv('./titanic_ml_dataset/Titanic_train.csv', sep=',')
print('titanic 변수 type:', type(titanic_df))
# titanic 변수 type: <class 'pandas.core.frame.DataFrame'>
print(titanic_df)
"""
     PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0              1         0       3  ...   7.2500   NaN         S
1              2         1       1  ...  71.2833   C85         C
2              3         1       3  ...   7.9250   NaN         S
3              4         1       1  ...  53.1000  C123         S
4              5         0       3  ...   8.0500   NaN         S
..           ...       ...     ...  ...      ...   ...       ...
886          887         0       2  ...  13.0000   NaN         S
887          888         1       1  ...  30.0000   B42         S
888          889         0       3  ...  23.4500   NaN         S
889          890         1       1  ...  30.0000  C148         C
890          891         0       3  ...   7.7500   NaN         Q

[891 rows x 12 columns]
"""

# head()와 tail(): 일부 데이터만 추출
print(titanic_df.head(3))
"""
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
"""

print(titanic_df.tail(3))
"""
     PassengerId  Survived  Pclass  ...   Fare Cabin  Embarked
888          889         0       3  ...  23.45   NaN         S
889          890         1       1  ...  30.00  C148         C
890          891         0       3  ...   7.75   NaN         Q
"""

# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.max_colwidth', 100)
# pd.set_option('display.max_columns', 100)

# shape
print('DataFrame 크기:', titanic_df.shape)
# DataFrame 크기: (891, 12)

# DataFrame의 생성
dic1 = {
    'Name': ['Soonhan', 'Kyuri', 'SoonKyu'],
    'Year': [1988, 1986, 2022],
    'Gender': ['Male', 'Female', 'Female']
}

# Dictonary를 DataFrame으로 변환
data_df = pd.DataFrame(dic1)
print(data_df)
print('#'*30)
"""
      Name  Year  Gender
0  Soonhan  1988    Male
1    Kyuri  1986  Female
2  SoonKyu  2022  Female
"""

## 새로운 컬럼명을 추가
data_df = pd.DataFrame(dic1, columns=['Name', 'Year', 'Gender', 'Age'])
print(data_df)
print('#'*30)
"""
      Name  Year  Gender  Age
0  Soonhan  1988    Male  NaN
1    Kyuri  1986  Female  NaN
2  SoonKyu  2022  Female  NaN
"""

## 인덱스를 새로운 값으로 할당
data_df = pd.DataFrame(dic1, index=['one', 'two', 'three'])
print(data_df)
print('#'*30)
"""
          Name  Year  Gender
one    Soonhan  1988    Male
two      Kyuri  1986  Female
three  SoonKyu  2022  Female
"""

# DataFrame의 컬럼명과 인덱스
print('columns', titanic_df.columns)
print('index', titanic_df.index)
print('index value', titanic_df.index.values)
"""
columns Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
index RangeIndex(start=0, stop=891, step=1)
index value [  0   1   2 ... 890 ]
"""

# info()
print(titanic_df.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object
 5   Age          714 non-null    float64 # 891-714 = 177(NaN)
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None
"""

# decribe(): 평균, 표준편차, 4분위 분포도
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

