import pandas as pd

# Missing Data 처리하기
titanic_df = pd.read_csv('./titanic_ml_dataset/Titanic_train.csv', sep=',')

print(titanic_df.isna().head(3))
"""
   PassengerId  Survived  Pclass   Name    Sex    Age  SibSp  Parch  Ticket   Fare  Cabin  Embarked
0        False     False   False  False  False  False  False  False   False  False   True     False
1        False     False   False  False  False  False  False  False   False  False  False     False
2        False     False   False  False  False  False  False  False   False  False   True     False
"""
# 컬럼별로 NaN 건수 구하기
print(titanic_df.isna().sum())
"""
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
"""

# fillna()로 Missing 데이터 대체
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('COOO')
print(titanic_df.head(3))
"""
   PassengerId  Survived  Pclass                                               Name     Sex  ...  Parch            Ticket     Fare Cabin  Embarked
0            1         0       3                            Braund, Mr. Owen Harris    male  ...      0         A/5 21171   7.2500  COOO         S
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  ...      0          PC 17599  71.2833   C85         C
2            3         1       3                             Heikkinen, Miss. Laina  female  ...      0  STON/O2. 3101282   7.9250  COOO         S
"""

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
print(titanic_df.isna().sum())
"""
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Cabin          0
Embarked       0
dtype: int64
"""
