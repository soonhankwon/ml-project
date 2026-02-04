import pandas as pd
import numpy as np

# 넘파이 ndarray, 리스트, 딕셔너리를 DataFrame으로 변환하기
col_name1=['col1']
list1=[1,2,3]
array1 = np.array(list1)

print('array1 shape:', array1.shape)
# array1 shape: (3,)

df_list1 = pd.DataFrame(array1, columns=col_name1)
print('1차원 리스트로 만든 DataFrame:\n', df_list1)
"""
1차원 리스트로 만든 DataFrame:
    col1
0     1
1     2
2     3
"""
df_array1 = pd.DataFrame(array1, columns=col_name1)
print('1차원 ndarray로 만든 DataFrame:\n', df_list1)
"""
1차원 ndarray로 만든 DataFrame:
    col1
0     1
1     2
2     3
"""

print(df_list1.shape)
# (3, 1)

col_name2=['col1', 'col2', 'col3']
list2=[[1,2,3], [11,12,13]]
array2 = np.array(list2)
print('array2 shape:', array2.shape)
# array2 shape: (2, 3)

df_list2 = pd.DataFrame(list2, columns=col_name2)
print('2차원 리스트로 만든 DataFrame:\n', df_list2)
"""
2차원 리스트로 만든 DataFrame:
    col1  col2  col3
0     1     2     3
1    11    12    13
"""

df_array2 = pd.DataFrame(array2, columns=col_name2)
print('2치원 ndarray로 만든 DataFrame:\n', df_array2)
"""
2치원 ndarray로 만든 DataFrame:
    col1  col2  col3
0     1     2     3
1    11    12    13
"""

dict = {'col1': [1,11], 'col2': [2,22], 'col3': [3,33]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame:\n', df_dict)
"""
딕셔너리로 만든 DataFrame:
    col1  col2  col3
0     1     2     3
1    11    22    33
"""

# DataFrame을 ndarray로 변환
array3 = df_dict.values
print('df_dict.values 타입:', type(array3), 'df_dic.values shape:', array3.shape)
print(array3)
"""
df_dict.values 타입: <class 'numpy.ndarray'> df_dic.values shape: (2, 3)
[[ 1  2  3]
 [11 22 33]]
"""
# DataFrame을 리스트로 변환
list3 = df_dict.values.tolist()
print('df_dict.values.tolist() 타입:', type(list3))
print(list3)
"""
df_dict.values.tolist() 타입: <class 'list'>
[[1, 2, 3], [11, 22, 33]]
"""
# DataFrame을 딕셔너리로 변환
dict3 = df_dict.to_dict('list')
print('df_dict.to_dict() 타입:', type(dict3))
print(dict3)
"""
df_dict.to_dict() 타입: <class 'dict'>
{'col1': [1, 11], 'col2': [2, 22], 'col3': [3, 33]}
"""

# DataFrame의 컬럼 데이터 세트 생성과 수정
titanic_df = pd.read_csv('./titanic_ml_dataset/Titanic_train.csv', sep=',')
titanic_df['Age_0'] = 0
print(titanic_df.head())
"""
   PassengerId  Survived  Pclass  ... Cabin Embarked  Age_0
0            1         0       3  ...   NaN        S      0
1            2         1       1  ...   C85        C      0
2            3         1       3  ...   NaN        S      0
3            4         1       1  ...  C123        S      0
4            5         0       3  ...   NaN        S      0
"""

titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch']+1
print(titanic_df.head(3))
"""
   PassengerId  Survived  Pclass  ... Cabin Embarked  Age_0
0            1         0       3  ...   NaN        S      0
1            2         1       1  ...   C85        C      0
2            3         1       3  ...   NaN        S      0
3            4         1       1  ...  C123        S      0
4            5         0       3  ...   NaN        S      0

[5 rows x 13 columns]
   PassengerId  Survived  Pclass  ... Age_0 Age_by_10  Family_No
0            1         0       3  ...     0     220.0          2
1            2         1       1  ...     0     380.0          2
2            3         1       3  ...     0     260.0          1
3            4         1       1  ...     0     350.0          2
4            5         0       3  ...     0     350.0          1

[5 rows x 15 columns]
"""

titanic_df['Age_by_10'] = titanic_df['Age_by_10'] + 100
print(titanic_df.head(3))
"""
   PassengerId  Survived  Pclass  ... Age_0 Age_by_10  Family_No
0            1         0       3  ...     0     320.0          2
1            2         1       1  ...     0     480.0          2
2            3         1       3  ...     0     360.0          1

[3 rows x 15 columns]
"""

# axis에 따른 삭제
titanic_drop_df = titanic_df.drop('Age_0', axis=1)
print(titanic_drop_df.head(3))
"""
   PassengerId  Survived  Pclass  ... Embarked Age_by_10  Family_No
0            1         0       3  ...        S     320.0          2
1            2         1       1  ...        C     480.0          2
2            3         1       3  ...        S     360.0          1
"""


drop_result = titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis=1, inplace=True)
print('inplace=True로 drop 후 반환된 값:', drop_result)
print(titanic_df.head(3))
"""
inplace=True로 drop 후 반환된 값: None <- inplace=True 경우 titanic_df 자체가 바뀜
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
"""

# axis=0 일 경우 drop()은 row 방향으로 데이터를 삭제
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 15)
print('### before axis 0 drop ###')
print(titanic_df.head(6))

"""
### before axis 0 drop ###
   PassengerId  Survived  Pclass            Name     Sex  ...  Parch          Ticket     Fare Cabin  Embarked
0            1         0       3  Braund, Mr....    male  ...      0       A/5 21171   7.2500   NaN         S
1            2         1       1  Cumings, Mr...  female  ...      0        PC 17599  71.2833   C85         C
2            3         1       3  Heikkinen, ...  female  ...      0  STON/O2. 31...   7.9250   NaN         S
3            4         1       1  Futrelle, M...  female  ...      0          113803  53.1000  C123         S
4            5         0       3  Allen, Mr. ...    male  ...      0          373450   8.0500   NaN         S
5            6         0       3  Moran, Mr. ...    male  ...      0          330877   8.4583   NaN 
"""

titanic_df.drop([0,1,2], axis=0, inplace=True)
print(titanic_df.head(3))
"""
   PassengerId  Survived  Pclass            Name     Sex  ...  Parch  Ticket     Fare Cabin  Embarked
3            4         1       1  Futrelle, M...  female  ...      0  113803  53.1000  C123         S
4            5         0       3  Allen, Mr. ...    male  ...      0  373450   8.0500   NaN         S
5            6         0       3  Moran, Mr. ...    male  ...      0  330877   8.4583   NaN         Q
"""