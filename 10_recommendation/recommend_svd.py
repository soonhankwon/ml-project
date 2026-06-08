# Surprise를 이용한 추천 시스템 구축

from pandas.core.common import random_state
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# 내장 데이터를 로드하고 학습과 테스트 데이터로 분리
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.25, random_state=0)

# 추천 행렬 분해 알고리즘으로 SVD 객체를 생성하고 학습 수행
algo = SVD()
algo.fit(trainset)

# 테스트 데이터 세트에 예상 평점 데이터 예측, test()메서드 호출시 Prediction 객체의 리스트로 평점 예측 데이터 반환
predictions = algo.test(testset)
print('prediction type :', type(predictions), ' size:',len(predictions))
print('prediction 결과의 최초 5개 추출')

"""
prediction type : <class 'list'>  size: 25000
prediction 결과의 최초 5개 추출
"""
print(predictions[:5])
print([ (pred.uid, pred.iid, pred.est) for pred in predictions[:3] ])
"""
[Prediction(uid='120', iid='282', r_ui=4.0, est=np.float64(3.757927079852352), details={'was_impossible': False}), 
Prediction(uid='882', iid='291', r_ui=4.0, est=np.float64(3.9232748006086675), details={'was_impossible': False}), 
Prediction(uid='535', iid='507', r_ui=5.0, est=np.float64(4.126591498092467), details={'was_impossible': False}), 
Prediction(uid='697', iid='244', r_ui=5.0, est=np.float64(3.6043677933049088), details={'was_impossible': False}), 
Prediction(uid='751', iid='385', r_ui=4.0, est=np.float64(3.6054960627179993), details={'was_impossible': False})]

[('120', '282', np.float64(3.757927079852352)), 
('882', '291', np.float64(3.9232748006086675)), 
('535', '507', np.float64(4.126591498092467))]
"""

# predict() 메서드는 개별 사용자, 아이템에 대한 예측 평점 정보를 반환
# 사용자 아이디, 아이템 아이디는 문자열로 입력해야 함
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid)
print(pred)
"""
user: 196        item: 302        r_ui = None   est = 4.08   {'was_impossible': False}
"""

# 반환된 Prediction의 리스트 객체를 기반으로 RMSE 평가
print(accuracy.rmse(predictions))
"""
RMSE: 0.9474
0.9473638204045097
"""

# Surprise 주요 모듈 소개
import pandas as pd
# csv 파일로 사용자 평점 데이터 생성
ratings = pd.read_csv('./ml-latest-small/ratings.csv')
# ratings_noh.csv 파일로 unload 시 index 와 header를 모두 제거한 새로운 파일 생성.  
ratings.to_csv('./ml-latest-small/ratings_noh.csv', index=False, header=False)

# Reader 클래스로 파일의 포맷팅 지정하고 Dataset의 load_from_file()을 이용하여 데이터셋 로딩
from surprise import Reader

reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
data = Dataset.load_from_file('./ml-latest-small/ratings_noh.csv', reader=reader)

# 학습과 테스트 데이터 세트로 분할하고 SVD로 학습후 테스트 데이터 평점 예측 후 RMSE 평가
trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD(n_factors=50, random_state=0)

# 학습 데이터 세트로 학습 후 테스트 데이터 세트로 평점 예측 후 RMSE 평가
algo.fit(trainset)
predictions = algo.test(testset)
print(accuracy.rmse(predictions))
"""
RMSE: 0.8682
0.8681952927143516
"""

# 판다스 DataFrame 기반에서 동일하게 재수행
ratings = pd.read_csv('./ml-latest-small/ratings.csv')
reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD(n_factors=50, random_state=0)

algo.fit(trainset)
predictions = algo.test(testset)
print(accuracy.rmse(predictions))
"""
RMSE: 0.8682
0.8681952927143516
"""

# 교차 검증(Cross Validation)과 하이퍼 파라미터 튜닝
# cross_validate()를 이용한 교차 검증
from surprise.model_selection import cross_validate
ratings = pd.read_csv('./ml-latest-small/ratings.csv')
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

algo = SVD(random_state=0)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
"""
Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8737  0.8728  0.8678  0.8685  0.8803  0.8726  0.0045  
MAE (testset)     0.6711  0.6694  0.6648  0.6665  0.6786  0.6701  0.0048  
Fit time          0.29    0.31    0.29    0.30    0.31    0.30    0.01    
Test time         0.05    0.05    0.05    0.05    0.07    0.05    0.01
"""

# GridSearchCV 이용
from surprise.model_selection import GridSearchCV

# 최적화할 파라미터를 딕셔너리 형태로 지정.
param_grid = {'n_epochs': [20, 40, 60], 'n_factors': [50, 100, 200]}

# CV를 3개 폴드 세트로 지정, 성능 평가는 rmse, mse로 수행하도록 GridSearchCV 구성
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

# 최고 RMSE Evaluation 점수와 그때의 하이퍼 파라미터
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
"""
0.8786364870216726
{'n_epochs': 20, 'n_factors': 50}
"""

# Surprise를 이용한 개인화 영화 추천 시스템 구축
# DatasetAutoFolds를 이용한 전체 데이터를 TrainSet 클래스 변환
from surprise.dataset import DatasetAutoFolds

reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
# DatasetAutoFolds 클래스를 rating_noh.csv 파일 기반으로 생성
data_folds = DatasetAutoFolds(ratings_file='./ml-latest-small/ratings_noh.csv', reader=reader)

# 전체 데이터를 학습 데이터로 생성함
trainset = data_folds.build_full_trainset()

# SVD로 학습
algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)

# 영화에 대한 상세 속성 정보 DataFrame 로딩
movies = pd.read_csv('./ml-latest-small/movies.csv')

movieIds = ratings[ratings['userId'] == 9]['movieId']
if movieIds[movieIds == 42].count() == 0:
    print('사용자 아이디 9는 영화 아이디 42의 평점 없음')
print(movies[movies['movieId']==42])
"""
사용자 아이디 9는 영화 아이디 42의 평점 없음
    movieId                   title              genres
38       42  Dead Presidents (1995)  Action|Crime|Drama
user: 9          item: 42         r_ui = None   est = 3.13   {'was_impossible': False}
"""

uid = str(9)
iid = str(42)

pred = algo.predict(uid, iid, verbose=True)

def get_unseen_surprise(ratings, movies, userId):
    #입력값으로 들어온 userId에 해당하는 사용자가 평점을 매긴 모든 영화를 리스트로 생성
    seen_movies = ratings[ratings['userId']== userId]['movieId'].tolist()
    
    # 모든 영화들의 movieId를 리스트로 생성. 
    total_movies = movies['movieId'].tolist()
    
    # 모든 영화들의 movieId중 이미 평점을 매긴 영화의 movieId를 제외하여 리스트로 생성
    unseen_movies= [movie for movie in total_movies if movie not in seen_movies]
    print('평점 매긴 영화수:',len(seen_movies), '추천대상 영화수:',len(unseen_movies), \
          '전체 영화수:',len(total_movies))
    
    return unseen_movies

unseen_movies = get_unseen_surprise(ratings, movies, 9)
"""
평점 매긴 영화수: 46 추천대상 영화수: 9696 전체 영화수: 9742
"""

def recomm_movie_by_surprise(algo, userId, unseen_movies, top_n=10):
    # 알고리즘 객체의 predict() 메서드를 평점이 없는 영화에 반복 수행한 후 결과를 list 객체로 저장
    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]
    
    # predictions list 객체는 surprise의 Predictions 객체를 원소로 가지고 있음.
    # [Prediction(uid='9', iid='1', est=3.69), Prediction(uid='9', iid='2', est=2.98),,,,]
    # 이를 est 값으로 정렬하기 위해서 아래의 sortkey_est 함수를 정의함.
    # sortkey_est 함수는 list 객체의 sort() 함수의 키 값으로 사용되어 정렬 수행.
    def sortkey_est(pred):
        return pred.est
    
    # sortkey_est( ) 반환값의 내림 차순으로 정렬 수행하고 top_n개의 최상위 값 추출.
    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions= predictions[:top_n]
    
    # top_n으로 추출된 영화의 정보 추출. 영화 아이디, 추천 예상 평점, 제목 추출
    top_movie_ids = [ int(pred.iid) for pred in top_predictions]
    top_movie_rating = [ pred.est for pred in top_predictions]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)]['title']
    top_movie_preds = [ (id, title, rating) for id, title, rating in zip(top_movie_ids, top_movie_titles, top_movie_rating)]
    
    return top_movie_preds

top_movie_preds = recomm_movie_by_surprise(algo, 9, unseen_movies, top_n=10)
print('##### Top-10 추천 영화 리스트 #####')

for top_movie in top_movie_preds:
    print(top_movie[1], ":", top_movie[2])

"""
##### Top-10 추천 영화 리스트 #####
Usual Suspects, The (1995) : 4.306302135700814
Star Wars: Episode IV - A New Hope (1977) : 4.281663842987387
Pulp Fiction (1994) : 4.278152632122758
Silence of the Lambs, The (1991) : 4.226073566460876
Godfather, The (1972) : 4.1918097904381995
Streetcar Named Desire, A (1951) : 4.154746591122658
Star Wars: Episode V - The Empire Strikes Back (1980) : 4.122016128534504
Star Wars: Episode VI - Return of the Jedi (1983) : 4.108009609093436
Goodfellas (1990) : 4.083464936588478
Glory (1989) : 4.07887165526957
"""