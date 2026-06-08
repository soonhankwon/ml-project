import numpy as np
from sklearn.metrics import mean_squared_error

def get_rmse(R, P, Q, non_zeros):
    error = 0
    # 두개의 분해된 행렬 P와 Q.T의 내적 곱으로 예측 R행렬 생성
    full_pred_matrix = np.dot(P, Q.T)

    # 실제 R행렬에서 null이 아닌 값의 위치 인덱스 추출하여 실제 R행렬과 예측 행렬 RMSE 추출
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]

    full_pred_matrix_non_zeroes = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]

    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeroes)
    rmse = np.sqrt(mse)
    return rmse

# 경사하강법에 기반하여 P와 Q의 원소들을 업데이트 수행
def matrix_fatorization(R, K, steps=20, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape
    # P와 Q 매트릭스의 크기를 지정하고 정규분포를 가진 랜덤한 값으로 입력
    np.random.seed(1)
    P = np.random.normal(scale=1./K, size=(num_users, K))
    Q = np.random.normal(scale=1./K, size=(num_items, K))

    # R>0 인 행 위치, 열 위치, 값을 non_zeroes 리스트 객체에 저장
    non_zeros = [(i, j, R[i,j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

    # SGD기법으로 P와 Q 매트릭스를 계속 업데이트. 
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구함
            eij = r - np.dot(P[i, :], Q[j, :].T)
            # Regularization을 반영한 SGD 업데이트 공식 적용
            P[i,:] = P[i,:] + learning_rate*(eij * Q[j, :] - r_lambda*P[i,:])
            Q[j,:] = Q[j,:] + learning_rate*(eij * P[i, :] - r_lambda*Q[j,:])
       
        rmse = get_rmse(R, P, Q, non_zeros)
        if (step % 10) == 0:
            print("### iteration step : ", step, " rmse : ", rmse)
            
    return P, Q

import pandas as pd

movies = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')
ratings_matrix = ratings.pivot_table('rating', index='userId', columns='movieId')

# title 컬럼을 얻기 위해 movies와 조인 수행
rating_movies = pd.merge(ratings, movies, on='movieId')

# columns=title로 title 컬럼으로 pivot 수행
ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')

P, Q = matrix_fatorization(ratings_matrix.values, K=50, steps=200, learning_rate=0.01, r_lambda=0.01)
pred_matrix = np.dot(P, Q.T)
"""
### iteration step :  0  rmse :  2.9023619751336867
### iteration step :  10  rmse :  0.7335768591017926
### iteration step :  20  rmse :  0.5115539026853442
### iteration step :  30  rmse :  0.37261628282537446
### iteration step :  40  rmse :  0.2960818299181014
### iteration step :  50  rmse :  0.2520353192341642
### iteration step :  60  rmse :  0.22487503275269854
### iteration step :  70  rmse :  0.2068545530233154
### iteration step :  80  rmse :  0.19413418783028685
### iteration step :  90  rmse :  0.18470082002720403
### iteration step :  100  rmse :  0.17742927527209104
### iteration step :  110  rmse :  0.17165226964707492
### iteration step :  120  rmse :  0.16695181946871723
### iteration step :  130  rmse :  0.16305292191997542
### iteration step :  140  rmse :  0.15976691929679643
### iteration step :  150  rmse :  0.1569598699945732
### iteration step :  160  rmse :  0.15453398186715428
### iteration step :  170  rmse :  0.15241618551077643
### iteration step :  180  rmse :  0.15055080739628307
### iteration step :  190  rmse :  0.14889470913232092
"""

ratings_pred_matrix = pd.DataFrame(data=pred_matrix, index= ratings_matrix.index, \
columns = ratings_matrix.columns)

print(ratings_pred_matrix.head(3))
"""
### iteration step :  190  rmse :  0.14889470913232092
title   '71 (2014)  ...  À nous la liberté (Freedom for Us) (1931)
userId              ...                                           
1         3.055084  ...                                   0.859474
2         3.170119  ...                                   0.725684
3         2.307073  ...                                   0.396941

[3 rows x 9719 columns]
"""

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
recomm_movies = pd.DataFrame(data=recomm_movies.values, index=recomm_movies.index, \
columns=['pred_score'])
print(recomm_movies.head(10))
"""
                                                    pred_score
title                                                         
Rear Window (1954)                                    5.704612
South Park: Bigger, Longer and Uncut (1999)           5.451100
Rounders (1998)                                       5.298393
Blade Runner (1982)                                   5.244951
Roger & Me (1989)                                     5.191962
Gattaca (1997)                                        5.183179
Ben-Hur (1959)                                        5.130463
Rosencrantz and Guildenstern Are Dead (1990)          5.087375
Big Lebowski, The (1998)                              5.038690
Star Wars: Episode V - The Empire Strikes Back ...    4.989601
"""