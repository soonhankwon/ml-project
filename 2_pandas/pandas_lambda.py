import pandas as pd
import numpy as np

# 파이썬 lambda식 기본
lambda_square = lambda x: x ** 2
print('3의 제곱은:', lambda_square(3)) # 3의 제곱은: 9

a = [1,2,3]
squares = map(lambda x: x ** 2, a)
print(list(squares)) # [1, 4, 9]

# 판다스에 apply lambda 적용
titanic_df = pd.read_csv('./titanic_ml_dataset/Titanic_train.csv', sep=',')
titanic_df['Name_len'] = titanic_df['Name'].apply(lambda x: len(x))
print(titanic_df[['Name', 'Name_len']].head(3))
"""
                                                Name  Name_len
0                            Braund, Mr. Owen Harris        23
1  Cumings, Mrs. John Bradley (Florence Briggs Th...        51
2                             Heikkinen, Miss. Laina        22
"""

titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x: 'Child' if x <= 15 else 'Adult')
print(titanic_df[['Age', 'Child_Adult']].head(10))
"""
    Age Child_Adult
0  22.0       Adult
1  38.0       Adult
2  26.0       Adult
3  35.0       Adult
4  35.0       Adult
5   NaN       Adult
6  54.0       Adult
7   2.0       Child
8  27.0       Adult
9  14.0       Child
"""

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: 'Child' if x <= 15 else ('Adult' if x <= 60 else 'Elderly'))
print(titanic_df['Age_cat'].value_counts())
"""
Age_cat
Adult      609
Elderly    199
Child       83
"""

def get_category(age):
    cat = ''
    if age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else: cat = 'Elderly'

    return cat

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: get_category(x))
print(titanic_df[['Age', 'Age_cat']].head())
"""
    Age      Age_cat
0  22.0      Student
1  38.0        Adult
2  26.0  Young Adult
3  35.0  Young Adult
4  35.0  Young Adult
"""