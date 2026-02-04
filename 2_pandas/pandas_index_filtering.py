import pandas as pd
from pandas.core import series

titanic_df = pd.read_csv('./titanic_ml_dataset/Titanic_train.csv', sep=',')

# DataFrame 객체에서 []연산자내에 한개의 컬럼만 입력하면 Series 객체를 반환
series = titanic_df['Name']
print(series.head(3))
print('## type:', type(series), 'shape:', series.shape)

# DataFrame 객체에서 []연산자내에 여러개의 컬럼을 리스트로 입력하면 그 컬럼들로 구성된 DataFrame 반환
filtered_df = titanic_df[['Name', 'Age']]
print(filtered_df.head(3))
print('## type:', type(filtered_df), 'shape:', filtered_df.shape)
"""
0                              Braund, Mr. Owen Harris
1    Cumings, Mrs. John Bradley (Florence Briggs Th...
2                               Heikkinen, Miss. Laina
Name: Name, dtype: str
## type: <class 'pandas.Series'> shape: (891,)
                                                Name   Age
0                            Braund, Mr. Owen Harris  22.0
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  38.0
2                             Heikkinen, Miss. Laina  26.0
## type: <class 'pandas.DataFrame'> shape: (891, 2)
"""

# DataFrame 객체에서 []연산자내에 한개의 컬럼을 리스트로 입력하면 한개의 컬럼으로 구성된 DataFrame 반환
one_col_df = titanic_df[['Name']] # 명확하게 2차원 명시
print(one_col_df.head(3))
print('## type:', type(one_col_df), 'shape:', one_col_df.shape)
"""
                                                Name
0                            Braund, Mr. Owen Harris
1  Cumings, Mrs. John Bradley (Florence Briggs Th...
2                             Heikkinen, Miss. Laina
## type: <class 'pandas.DataFrame'> shape: (891, 1)
"""

# print('[]안에 숫자 index는 KeyError 오류 발생:\n', titanic_df[0])
print(titanic_df[0:2])
"""
   PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch     Ticket     Fare Cabin Embarked
0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0  A/5 21171   7.2500   NaN        S
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0   PC 17599  71.2833   C85        C
"""

print(titanic_df[titanic_df['Pclass'] == 3].head(3))
"""
   PassengerId  Survived  Pclass                      Name     Sex   Age  SibSp  Parch            Ticket   Fare Cabin Embarked
0            1         0       3   Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171  7.250   NaN        S
2            3         1       3    Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282  7.925   NaN        S
4            5         0       3  Allen, Mr. William Henry    male  35.0      0      0            373450  8.050   NaN        S
"""

# DataFrame iloc[] 연산자
data = {
    'Name': ['Soonhan', 'Kyuri', 'Soonkyu'],
    'Year': [1988, 1986, 2022],
    'Gender': ['Male', 'Female', 'Male']
}

data_df = pd.DataFrame(data, index=['one', 'two', 'three'])
print(data_df)
print(data_df.iloc[0,0])
"""
          Name  Year  Gender
one    Soonhan  1988    Male
two      Kyuri  1986  Female
three  Soonkyu  2022    Male

Soonhan
"""

# data.df.iloc[0, 'Name'] 오류발생 코드
# data_df.iloc['one', 0] 오류발생 코드
print("\niloc[1,0] 두번째 행의 첫번째 열 값:", data_df.iloc[1,0])
print("\niloc[2,1] 세번째 행의 두번째 열 값:", data_df.iloc[2,1])
"""
iloc[1,0] 두번째 행의 첫번째 열 값: Kyuri

iloc[2,1] 세번째 행의 두번째 열 값: 2022
"""

print("\niloc[0:2, [0,1]] 첫번째에서 두번째 행의 첫번째, 두번째 열 값:\n", data_df.iloc[0:2, [0,1]])
print("\niloc[0:2, 0:3] 첫번째에서 두번째 행의 첫번째부터 세번째 열 값:\n", data_df.iloc[0:2, 0:3])
"""
iloc[0:2, [0,1]] 첫번째에서 두번째 행의 첫번째, 두번째 열 값:
         Name  Year
one  Soonhan  1988
two    Kyuri  1986

iloc[0:2, 0:3] 첫번째에서 두번째 행의 첫번째부터 세번째 열 값:
         Name  Year  Gender
one  Soonhan  1988    Male
two    Kyuri  1986  Female
"""
print("\n맨 마지막 칼럼 데이터 [:, -1] \n", data_df.iloc[:, -1])
print("\n맨 마지막 칼럼을 제외한 모든 데이터 [:, :-1] \n", data_df.iloc[:, :-1])
"""
맨 마지막 칼럼 데이터 [:, -1] 
one        Male
two      Female
three      Male
Name: Gender, dtype: str

 맨 마지막 칼럼을 제외한 모든 데이터 [:, :-1] 
           Name  Year
one    Soonhan  1988
two      Kyuri  1986
three  Soonkyu  2022
"""
# iloc[]는 불린 인덱싱을 지원하지 않는다.

# DataFrame loc[] 연산자
print(data_df.loc['one', 'Name']) # Soonhan
# data_df.loc[0, 'Name'] 오류 발생(현재 인덱스는 one, two, three)

print('위치기반 iloc slicing\n', data_df.iloc[0:1, 0], '\n')
print('명칭기반 loc slicing\n', data_df.loc['one':'two', 'Name']) # two 인덱스는(-)가 되지않음으로 포함시킴
"""
위치기반 iloc slicing
 one    Soonhan
Name: Name, dtype: str 

명칭기반 loc slicing
 one    Soonhan
two      Kyuri
Name: Name, dtype: str
"""

# 불린 인덱싱
print('인덱스 값 three인 행의 Name 칼럼값:', data_df.loc['three', 'Name'])
print('\n인덱스 값 one부터 two까지 행의 Name과 Year 칼럼값:\n', data_df.loc['one':'two', ['Name', 'Year']])
print('\n인덱스 값 one부터 three까지 행의 Name부터 Gender까지의 칼럼값:\n', data_df.loc['one':'three', 'Name':'Gender'])
print('\n모든 데이터 값:\n', data_df.loc[:])
print('\n불린 인덱싱: \n', data_df.loc[data_df.Year >= 2014])

"""
인덱스 값 three인 행의 Name 칼럼값: Soonkyu

인덱스 값 one부터 two까지 행의 Name과 Year 칼럼값:
         Name  Year
one  Soonhan  1988
two    Kyuri  1986

인덱스 값 one부터 three까지 행의 Name부터 Gender까지의 칼럼값:
           Name  Year  Gender
one    Soonhan  1988    Male
two      Kyuri  1986  Female
three  Soonkyu  2022    Male

모든 데이터 값:
           Name  Year  Gender
one    Soonhan  1988    Male
two      Kyuri  1986  Female
three  Soonkyu  2022    Male

불린 인덱싱: 
           Name  Year Gender
three  Soonkyu  2022   Male
"""

titanic_boolean = titanic_df[titanic_df['Age'] > 60]
print(type(titanic_boolean)) # <class 'pandas.DataFrame'>
print(titanic_boolean)
"""
     PassengerId  Survived  Pclass                                       Name     Sex  ...  Parch       Ticket      Fare        Cabin  Embarked
33            34         0       2                      Wheadon, Mr. Edward H    male  ...      0   C.A. 24579   10.5000          NaN         S
54            55         0       1             Ostby, Mr. Engelhart Cornelius    male  ...      1       113509   61.9792          B30         C
...
851          852         0       3                        Svensson, Mr. Johan    male  ...      0       347060    7.7750          NaN         S
"""

print(titanic_df[titanic_df['Age'] > 60][['Name', 'Age']].head(3))
"""
                              Name   Age
33           Wheadon, Mr. Edward H  66.0
54  Ostby, Mr. Engelhart Cornelius  65.0
96       Goldschmidt, Mr. George B  71.0
"""

print(titanic_df.loc[titanic_df['Age'] > 60, ['Name', 'Age']].head(3))
"""
                              Name   Age
33           Wheadon, Mr. Edward H  66.0
54  Ostby, Mr. Engelhart Cornelius  65.0
96       Goldschmidt, Mr. George B  71.0
"""

print(titanic_df[(titanic_df['Age'] > 60) & (titanic_df['Pclass'] ==1) & (titanic_df['Sex'] == 'female')])
"""
     PassengerId  Survived  Pclass                                       Name     Sex   Age  SibSp  Parch  Ticket     Fare Cabin Embarked
275          276         1       1          Andrews, Miss. Kornelia Theodosia  female  63.0      1      0   13502  77.9583    D7        S
829          830         1       1  Stone, Mrs. George Nelson (Martha Evelyn)  female  62.0      0      0  113572  80.0000   B28      NaN
"""

cond1 = titanic_df['Age'] > 60
cond2 = titanic_df['Pclass'] == 1
cond3 = titanic_df['Sex'] == 'female'
print(titanic_df[cond1 & cond2 & cond3])
"""
     PassengerId  Survived  Pclass                                       Name     Sex   Age  SibSp  Parch  Ticket     Fare Cabin Embarked
275          276         1       1          Andrews, Miss. Kornelia Theodosia  female  63.0      1      0   13502  77.9583    D7        S
829          830         1       1  Stone, Mrs. George Nelson (Martha Evelyn)  female  62.0      0      0  113572  80.0000   B28      NaN
"""