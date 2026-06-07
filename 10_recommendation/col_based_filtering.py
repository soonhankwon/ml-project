import pandas as pd
import numpy as np

movies = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

print(movies.shape)
print(ratings.shape)
print(movies.head())
print(ratings.head())
"""
(9742, 3)
(100836, 4)

   movieId                               title                                       genres
0        1                    Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy
1        2                      Jumanji (1995)                   Adventure|Children|Fantasy
2        3             Grumpier Old Men (1995)                               Comedy|Romance
3        4            Waiting to Exhale (1995)                         Comedy|Drama|Romance
4        5  Father of the Bride Part II (1995)

   userId  movieId  rating  timestamp
0       1        1     4.0  964982703
1       1        3     4.0  964981247
2       1        6     4.0  964982224
3       1       47     5.0  964983815
4       1       50     5.0  964982931
"""

# 로우레벨 사용자 평점 데이터를 사용자-아이템 평점 행렬로 변환
ratings = ratings[['userId', 'movieId', 'rating']]
ratings_matrix = ratings.pivot_table('rating', index='userId', columns='movieId')
print(ratings_matrix.head(3))
"""
movieId  1       2       3       4       5       6       7       ...  193573  193579  193581  193583  193585  193587  193609
userId                                                           ...                                                        
1           4.0     NaN     4.0     NaN     NaN     4.0     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN
2           NaN     NaN     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN
3           NaN     NaN     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN
"""

# title 컬럼을 얻기 위해 movies와 조인
rating_movies = pd.merge(ratings, movies, on='movieId')

# columns=title로 title 컬럼으로 pivot 수행
ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')

# NaN 값을 모두 0으로 변환
ratings_matrix = ratings_matrix.fillna(0)
print(ratings_matrix.head(3))
"""
title   '71 (2014)  'Hellboy': The Seeds of Creation (2004)  ...  ¡Three Amigos! (1986)  À nous la liberté (Freedom for Us) (1931)
userId                                                       ...                                                                  
1              0.0                                      0.0  ...                    4.0                                        0.0
2              0.0                                      0.0  ...                    0.0                                        0.0
3              0.0                                      0.0  ...                    0.0                                        0.0

[3 rows x 9719 columns]
"""

# 영화와 영화들 간 유사도 산출
ratings_matrix_T = ratings_matrix.transpose()
print(ratings_matrix_T.head(3))
"""
title                                                                                 ...
'71 (2014)                               0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0 4.0
'Hellboy': The Seeds of Creation (2004)  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0 0.0
'Round Midnight (1986)                   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0 0.0

[3 rows x 610 columns]
"""

from sklearn.metrics.pairwise import cosine_similarity

item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

# cosine_similarity()로 반환된 넘파이 행렬을 영화명을 매핑하여 DataFrame으로 반환
item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns, \
columns=ratings_matrix.columns)

print(item_sim_df.shape)
print(item_sim_df.head(3))
"""
(9719, 9719)

title                                    '71 (2014)  ...  À nous la liberté (Freedom for Us) (1931)
title                                                ...                                           
'71 (2014)                                      1.0  ...                                        0.0
'Hellboy': The Seeds of Creation (2004)         0.0  ...                                        0.0
'Round Midnight (1986)                          0.0  ...                                        0.0

[3 rows x 9719 columns]
"""

print(item_sim_df['Godfather, The (1972)'].sort_values(ascending=False)[:6])
"""
title
Godfather, The (1972)                        1.000000
Godfather: Part II, The (1974)               0.821773
Goodfellas (1990)                            0.664841
One Flew Over the Cuckoo's Nest (1975)       0.620536
Star Wars: Episode IV - A New Hope (1977)    0.595317
Fargo (1996)                                 0.588614
Name: Godfather, The (1972), dtype: float64
"""

print(item_sim_df["Inception (2010)"].sort_values(ascending=False)[1:6])
"""
title
Dark Knight, The (2008)          0.727263
Inglourious Basterds (2009)      0.646103
Shutter Island (2010)            0.617736
Dark Knight Rises, The (2012)    0.617504
Fight Club (1999)                0.615417
Name: Inception (2010), dtype: float64
"""

# 아이템 기반 인접 이웃 협업 필터링으로 개인화된 영화 추천
def predict_rating(ratings_arr, item_sim_arr):
    ratings_pred = ratings_arr.dot(item_sim_arr) / np.array([np.abs(item_sim_arr).sum(axis=1)])
    return ratings_pred

ratings_pred = predict_rating(ratings_matrix.values, item_sim_df.values)
ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index=ratings_matrix.index, \
columns=ratings_matrix.columns)

print(ratings_pred_matrix.head(3))
"""
title   '71 (2014)  'Hellboy': The Seeds of Creation (2004)  ...  ¡Three Amigos! (1986)  À nous la liberté (Freedom for Us) (1931)
userId                                                       ...                                                                  
1         0.070345                                 0.577855  ...               0.292955                                   0.720347
2         0.018260                                 0.042744  ...               0.017563                                   0.000000
3         0.011884                                 0.030279  ...               0.010420                                   0.084501

[3 rows x 9719 columns]
"""

# 가중치 평점 부여뒤에 예측 성능 평가 MSE를 구함
from sklearn.metrics import mean_squared_error

# 사용자가 평점을 부여한 영화에 대해서만 예측 성능 평가 MSE 를 구함. 
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

print('아이템 기반 모든 인접 이웃 MSE: ', get_mse(ratings_pred, ratings_matrix.values))
"""
아이템 기반 모든 인접 이웃 MSE:  9.895354759094706
"""

# top-n 유사도를 가진 데이터들에 대해서만 예측 평점 계산
def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    # 사용자-아이템 평점 행렬 크기만큼 0으로 채운 예측 행렬 초기화
    pred = np.zeros(ratings_arr.shape)

    # 사용자-아이템 평점 행렬의 열 크기만큼 Loop 수행. 
    for col in range(ratings_arr.shape[1]):
        # 유사도 행렬에서 유사도가 큰 순으로 n개 데이터 행렬의 index 반환
        top_n_items = np.argsort(item_sim_arr[:, col])[:-n-1:-1]
        # 개인화된 예측 평점을 계산
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, top_n_items].dot(ratings_arr[row, top_n_items])
            pred[row, col] /= np.sum(np.abs(item_sim_arr[col, top_n_items]))        
    return pred

# top-n 유사도 기반의 예측 평점 및 MSE 계산
ratings_pred = predict_rating_topsim(ratings_matrix.values, item_sim_df.values, n=20)
print('아이템 기반 인접 TOP-20 이웃 MSE: ', get_mse(ratings_pred, ratings_matrix.values ))
"""
아이템 기반 인접 TOP-20 이웃 MSE:  3.694957479362603
"""

# 계산된 예측 평점 데이터는 DataFrame으로 재생성
ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index= ratings_matrix.index, \
columns = ratings_matrix.columns)

user_rating_id = ratings_matrix.loc[9, :]
print(user_rating_id[user_rating_id > 0].sort_values(ascending=False)[:10])
"""
title
Adaptation (2002)                                                                 5.0
Citizen Kane (1941)                                                               5.0
Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)    5.0
Producers, The (1968)                                                             5.0
Lord of the Rings: The Two Towers, The (2002)                                     5.0
Lord of the Rings: The Fellowship of the Ring, The (2001)                         5.0
Back to the Future (1985)                                                         5.0
Austin Powers in Goldmember (2002)                                                5.0
Minority Report (2002)                                                            4.0
Witness (1985)                                                                    4.0
Name: 9, dtype: float64
"""

# 사용자가 관람하지 않은 영화 중에서 아이템 기반의 인접 이웃 협업 필터링으로 영화 추천
def get_unseen_movies(ratings_matrix, userId):
    # userId로 입력받은 사용자의 모든 영화정보 추출하여 Series로 반환함. 
    # 반환된 user_rating 은 영화명(title)을 index로 가지는 Series 객체임. 
    user_rating = ratings_matrix.loc[userId,:]
    
    # user_rating이 0보다 크면 기존에 관람한 영화임. 대상 index를 추출하여 list 객체로 만듬
    already_seen = user_rating[ user_rating > 0].index.tolist()
    
    # 모든 영화명을 list 객체로 만듬. 
    movies_list = ratings_matrix.columns.tolist()
    
    # list comprehension으로 already_seen에 해당하는 movie는 movies_list에서 제외함. 
    unseen_list = [ movie for movie in movies_list if movie not in already_seen]
    
    return unseen_list

# 아이템 기반 유사도로 평점이 부여된 데이터 세트에서 해당 사용자가 관람하지 않은 영화들의 예측 평점이 가장 높은 영화를 추천
def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    # 예측 평점 DataFrame에서 사용자id index와 unseen_list로 들어온 영화명 컬럼을 추출하여
    # 가장 예측 평점이 높은 순으로 정렬함. 
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies
    
# 사용자가 관람하지 않는 영화명 추출   
unseen_list = get_unseen_movies(ratings_matrix, 9)

# 아이템 기반의 인접 이웃 협업 필터링으로 영화 추천 
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)

# 평점 데이타를 DataFrame으로 생성. 
recomm_movies = pd.DataFrame(data=recomm_movies.values,index=recomm_movies.index,columns=['pred_score'])
print(recomm_movies.head(10))
"""
title                                                         
Shrek (2001)                                          0.866202
Spider-Man (2002)                                     0.857854
Last Samurai, The (2003)                              0.817473
Indiana Jones and the Temple of Doom (1984)           0.816626
Matrix Reloaded, The (2003)                           0.800990
Harry Potter and the Sorcerer's Stone (a.k.a. H...    0.765159
Gladiator (2000)                                      0.740956
Matrix, The (1999)                                    0.732693
Pirates of the Caribbean: The Curse of the Blac...    0.689591
Lord of the Rings: The Return of the King, The ...    0.676711
"""