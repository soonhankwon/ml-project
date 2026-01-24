import pandas as pd
from pandas.core import series

titanic_df = pd.read_csv('./titanic_ml_dataset/Titanic_train.csv', sep=',')

# Index 객체
indexes = titanic_df.index
print(indexes)
print('index 객체 array값: \n', indexes.values)
"""
RangeIndex(start=0, stop=891, step=1)
index 객체 array값: 
 [  0   1   2   3   4 ... 890]
"""

print(type(indexes.values))
print(indexes.values.shape)
print(indexes[:5].values)
print(indexes[6])
"""
<class 'numpy.ndarray'>
(891,)
[0 1 2 3 4]
6
"""

# indexes[0] = 5 안됨, 인덱스자체에는 직접연산X(value에만)
# Series는 인덱스가 있지만, 1차원(2차원으로 혼돈하지 말것).
sereis_fair = titanic_df['Fare']
print('Fair Series max 값: ', sereis_fair.max())
print('Fair Series sum 값: ', sereis_fair.sum())
print('sum() Fair Series:', sum(sereis_fair))
print('Fair Series + 3:\n', (sereis_fair + 3).head(3))
"""
Fair Series max 값:  512.3292
Fair Series sum 값:  28693.9493
sum() Fair Series: 28693.9493
Fair Series + 3:
 0    10.2500
1    74.2833
2    10.9250
Name: Fare, dtype: float64
"""

titanic_reset_df = titanic_df.reset_index(inplace=False)
print(titanic_reset_df.head(3))
"""
   index  PassengerId  Survived  ...     Fare Cabin Embarked
0      0            1         0  ...   7.2500   NaN        S
1      1            2         1  ...  71.2833   C85        C
2      2            3         1  ...   7.9250   NaN        S
"""

print('### before reset_index ###')
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입과 shape:', type(value_counts), value_counts.shape)
"""
### before reset_index ###
Pclass
3    491
1    216
2    184
Name: count, dtype: int64
value_counts 객체 변수 타입과 shape: <class 'pandas.core.series.Series'> (3,)
"""

new_value_counts_01 = value_counts.reset_index(inplace=False)
print('### After reser_index ###')
print(new_value_counts_01)
print('new_value_count_01 객체 변수타입과 shape:', type(new_value_counts_01), new_value_counts_01.shape)
"""
### After reser_index ###
   Pclass  count
0       3    491
1       1    216
2       2    184
new_value_count_01 객체 변수타입과 shape: <class 'pandas.core.frame.DataFrame'> (3, 2)
"""

new_value_counts_02 = value_counts.reset_index(drop=True, inplace=False)
print('### After reset_index with drop ###')
print(new_value_counts_02)
print('new_value_counts_02 객체변수 타입과 shape:', type(new_value_counts_02), new_value_counts_02.shape)
"""
### After reset_index with drop ###
0    491
1    216
2    184
Name: count, dtype: int64
new_value_counts_02 객체변수 타입과 shape: <class 'pandas.core.series.Series'> (3,)
"""

new_value_counts_01 = titanic_df['Pclass'].value_counts().reset_index()
new_value_counts_01.rename(columns={'index': 'Pclass', 'Pclass': 'Pclass_count'})
print(new_value_counts_01)
