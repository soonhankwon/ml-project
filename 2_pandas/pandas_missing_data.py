import pandas as pd
import numpy as np

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

# nunique로 컬럼내 몇건의 고유값이 있는지 파악
print(titanic_df['Name'].value_counts())
print(titanic_df['Pclass'].nunique())
print(titanic_df['Survived'].nunique())
print(titanic_df['Name'].nunique())
"""
Name
Braund, Mr. Owen Harris                     1
Boulos, Mr. Hanna                           1
Frolicher-Stehli, Mr. Maxmillian            1
Gilinski, Mr. Eliezer                       1
Murdlin, Mr. Joseph                         1
                                           ..
Kelly, Miss. Anna Katherine "Annie Kate"    1
McCoy, Mr. Bernard                          1
Johnson, Mr. William Cahoone Jr             1
Keane, Miss. Nora A                         1
Dooley, Mr. Patrick                         1
Name: count, Length: 891, dtype: int64

3
2
891
"""

# replece로 원본 값을 특정값으로 대체
replace_test_df = pd.read_csv('./titanic_ml_dataset/Titanic_train.csv', sep=',')
replace_test_df['Sex'].replace('male', 'Man')
replace_test_df['Sex'] = replace_test_df['Sex'].replace({'male': 'Man', 'female': 'Woman'})
print(replace_test_df['Sex'].head(3))
"""
0      Man
1    Woman
2    Woman
"""
print(replace_test_df['Cabin'].value_counts(dropna=False))
"""
Cabin
NaN            687
C23 C25 C27      4
G6               4
B96 B98          4
C22 C26          3
              ... 
E34              1
C7               1
C54              1
E36              1
C148             1
"""
replace_test_df['Cabin'] = replace_test_df['Cabin'].replace(np.nan, 'C001')
print(replace_test_df['Cabin'].value_counts(dropna=False))
"""
Cabin
C001           687
C23 C25 C27      4
G6               4
B96 B98          4
C22 C26          3
              ... 
E34              1
C7               1
C54              1
E36              1
C148             1
"""