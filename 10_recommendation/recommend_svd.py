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
