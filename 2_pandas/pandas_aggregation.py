import pandas as pd

# DataFrame, Series의 정렬 - sort_values()
titanic_df = pd.read_csv('./titanic_ml_dataset/Titanic_train.csv', sep=',')

# 이름으로 정렬
titanic_sorted = titanic_df.sort_values(by='Name')
print(titanic_sorted.head(3))
"""
     PassengerId  Survived  Pclass                              Name     Sex  ...  Parch     Ticket   Fare Cabin  Embarked
845          846         0       3               Abbing, Mr. Anthony    male  ...      0  C.A. 5547   7.55   NaN         S
746          747         0       3       Abbott, Mr. Rossmore Edward    male  ...      1  C.A. 2673  20.25   NaN         S
279          280         1       3  Abbott, Mrs. Stanton (Rosa Hunt)  female  ...      1  C.A. 2673  20.25   NaN         S
"""

# Pclass와 Name으로 내림차순 정렬
titanic_sorted = titanic_df.sort_values(by=['Pclass', 'Name'], ascending=False)
print(titanic_sorted.head(3))
"""
     PassengerId  Survived  Pclass                             Name   Sex  ...  Parch    Ticket  Fare Cabin  Embarked
868          869         0       3      van Melkebeke, Mr. Philemon  male  ...      0    345777   9.5   NaN         S
153          154         0       3  van Billiard, Mr. Austin Blyler  male  ...      2  A/5. 851  14.5   NaN         S
282          283         0       3        de Pelsmaeker, Mr. Alfons  male  ...      0    345778   9.5   NaN         S
"""

# Aggregation 함수 적용
# DataFrame의 건수를 알고싶다면 count()보다는 shape
print(titanic_df.count())
"""
PassengerId    891
Survived       891
Pclass         891
Name           891
Sex            891
Age            714
SibSp          891
Parch          891
Ticket         891
Fare           891
Cabin          204
Embarked       889
dtype: int64
"""

print(titanic_df[['Age', 'Fare']].mean())
"""
Age     29.699118
Fare    32.204208
dtype: float64
"""

print(titanic_df[['Age', 'Fare']].sum())
print(titanic_df[['Age', 'Fare']].count())
"""
Age     21205.1700
Fare    28693.9493
dtype: float64
Age     714
Fare    891
dtype: int64
"""

# groupby() 이용하기
titanic_groupby = titanic_df.groupby('Pclass')
print(type(titanic_groupby)) # <class 'pandas.core.groupby.generic.DataFrameGroupBy'>

print(titanic_groupby[['Age', 'Fare']].count())
"""
        Age  Fare
Pclass           
1       186   216
2       173   184
3       355   491
"""

titanic_groupby = titanic_df.groupby('Pclass').count()
print(titanic_groupby)
"""
        PassengerId  Survived  Name  Sex  Age  SibSp  Parch  Ticket  Fare  Cabin  Embarked
Pclass                                                                                    
1               216       216   216  216  186    216    216     216   216    176       214
2               184       184   184  184  173    184    184     184   184     16       184
3               491       491   491  491  355    491    491     491   491     12       491
"""

titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId', 'Survived']].count()
print(titanic_groupby)
"""
        PassengerId  Survived
Pclass                       
1               216       216
2               184       184
3               491       491
"""

# 서로 다른 aggregation 메소드를 호출해야할때
titanic_groupby = titanic_df.groupby('Pclass')['Age'].agg([max, min])
print(titanic_groupby)
"""
         max   min
Pclass            
1       80.0  0.92
2       70.0  0.67
3       74.0  0.42
"""

# 서로 다른 컬럼에 서로 다른 aggregation 메소드를 적용할 경우 agg()내에 컬럼과 적용할 메소드를 Dict로 입력
agg_format={'Age':'max', 'SibSp':'sum', 'Fare':'mean'}
titanic_groupby = titanic_df.groupby('Pclass').agg(agg_format)
print(titanic_groupby)
"""
         Age  SibSp       Fare
Pclass                        
1       80.0     90  84.154687
2       70.0     74  20.662183
3       74.0    302  13.675550
"""

# agg내의 인자로 들어가는 Dict 객체에 동일한 Key를 가지는 두 개의 value가 있을 경우 마지막 value로 update됨
agg_format={'Age':'max', 'Age':'mean', 'Fare':'mean'}
titanic_groupby = titanic_df.groupby('Pclass').agg(agg_format)
print(titanic_groupby)
"""
              Age       Fare
Pclass                      
1       38.233441  84.154687
2       29.877630  20.662183
3       25.140620  13.675550
"""
# named group by
titanic_groupby = titanic_df.groupby(['Pclass']).agg(age_max=('Age', 'max'), age_mean=('Age', 'mean'), fare_mean=('Fare', 'mean'))
print(titanic_groupby)
"""
        age_max   age_mean  fare_mean
Pclass                               
1          80.0  38.233441  84.154687
2          70.0  29.877630  20.662183
3          74.0  25.140620  13.675550
"""

titanic_groupby = titanic_df.groupby('Pclass').agg(
    age_max = pd.NamedAgg(column='Age', aggfunc='max'),
    age_mean = pd.NamedAgg(column='Age', aggfunc='mean'),
    fare_mean = pd.NamedAgg(column='Fare', aggfunc='mean')
)
print(titanic_groupby)
"""
        age_max   age_mean  fare_mean
Pclass                               
1          80.0  38.233441  84.154687
2          70.0  29.877630  20.662183
3          74.0  25.140620  13.675550
"""