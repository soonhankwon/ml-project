# 텍스트 정규화
from sklearn.datasets import fetch_20newsgroups

news_data = fetch_20newsgroups(subset='all', random_state=156)
print(news_data.keys())
"""
dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])
"""

import pandas as pd

print('target 클래스의 값과 분포도 \n',pd.Series(news_data.target).value_counts().sort_index())
print('target 클래스의 이름들 \n',news_data.target_names)
"""
target 클래스의 값과 분포도 
 0     799
1     973
2     985
3     982
4     963
5     988
6     975
7     990
8     996
9     994
10    999
11    991
12    984
13    990
14    987
15    997
16    910
17    940
18    775
19    628
Name: count, dtype: int64


target 클래스의 이름들 
 ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
"""

# 학습과 테스트용 데이터 생성
print(news_data.data[0])
"""
From: egreen@east.sun.com (Ed Green - Pixel Cruncher)
Subject: Re: Observation re: helmets
Organization: Sun Microsystems, RTP, NC
Lines: 21
Distribution: world
Reply-To: egreen@east.sun.com
NNTP-Posting-Host: laser.east.sun.com

In article 211353@mavenry.altcit.eskimo.com, maven@mavenry.altcit.eskimo.com (Norman Hamer) writes:
> 
> The question for the day is re: passenger helmets, if you don't know for 
>certain who's gonna ride with you (like say you meet them at a .... church 
>meeting, yeah, that's the ticket)... What are some guidelines? Should I just 
>pick up another shoei in my size to have a backup helmet (XL), or should I 
>maybe get an inexpensive one of a smaller size to accomodate my likely 
>passenger? 

If your primary concern is protecting the passenger in the event of a
crash, have him or her fitted for a helmet that is their size.  If your
primary concern is complying with stupid helmet laws, carry a real big
spare (you can put a big or small head in a big helmet, but not in a
small one).

---
Ed Green, former Ninjaite |I was drinking last night with a biker,
  Ed.Green@East.Sun.COM   |and I showed him a picture of you.  I said,
DoD #0111  (919)460-8302  |"Go on, get to know her, you'll like her!"
 (The Grateful Dead) -->  |It seemed like the least I could do...
"""

# subset='train'으로 학습용(Train) 데이터만 추출, remove=('headers', 'footers', 'quotes')로 내용만 추출
train_news= fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), random_state=156)
X_train = train_news.data
y_train = train_news.target
print(type(X_train))
"""
<class 'list'>
"""

# subset='test'으로 테스트(Test) 데이터만 추출, remove=('headers', 'footers', 'quotes')로 내용만 추출
test_news= fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'),random_state=156)
X_test = test_news.data
y_test = test_news.target
print(f'학습 데이터 크기 {len(train_news.data)}, 테스트 데이터 크기 {len(test_news.data)}')
"""
학습 데이터 크기 11314, 테스트 데이터 크기 7532
"""

# 피처 벡터화 변환과 머신러닝 모델 학습/예측/평가
## 학습 데이터에 대해 fit()된 CountVectorizer를 이용해서 테스트 데이터를 피처 벡터화 해야함

from sklearn.feature_extraction.text import CountVectorizer

# Count Vectorization으로 feature extraction 변환 수행
cnt_vect = CountVectorizer()

cnt_vect.fit(X_train)
X_train_cnt_vect = cnt_vect.transform(X_train)

# 학습 데이터로 fit()된 CountVectorizer를 이용하여 테스트 데이터를 feature extraction 변환 수행
X_test_cnt_vect = cnt_vect.transform(X_test)

print('학습 데이터 Text의 CountVectorizer Shape:', X_train_cnt_vect.shape)
"""
학습 데이터 Text의 CountVectorizer Shape: (11314, 101631)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# LogisticRegression을 이용하여 학습/예측/평가 수행
lr_clf = LogisticRegression(solver='lbfgs', max_iter=100)
lr_clf.fit(X_train_cnt_vect, y_train)
pred = lr_clf.predict(X_test_cnt_vect)
print(f'CountVectorized Logistic Regression의 예측 정확도는 {accuracy_score(y_test, pred):.3f}')
"""
CountVectorized Logistic Regression의 예측 정확도는 0.605
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization 적용하여 학습 데이터셋과 테스트 데이터셋 변환
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(X_train)
X_train_tfidf_vect = tfidf_vect.transform(X_train)
X_test_tfidf_vect = tfidf_vect.transform(X_test)

# LogisticRegression을 이용하여 학습/예측/평가 수행
lr_clf = LogisticRegression(solver='lbfgs', max_iter=100)
lr_clf.fit(X_train_tfidf_vect, y_train)
pred = lr_clf.predict(X_test_tfidf_vect)
print(f'TF-IDF Logistic Regression의 예측 정확도는 {accuracy_score(y_test, pred):.3f}')
"""
TF-IDF Logistic Regression의 예측 정확도는 0.674
"""

# stop words 필터링을 추가하고 ngram을 기본(1,1)에서 (1,2)로 변경하여 Feature Vectorization 적용.
tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300)
tfidf_vect.fit(X_train)
X_train_tfidf_vect = tfidf_vect.transform(X_train)
X_test_tfidf_vect = tfidf_vect.transform(X_test)

# LogisticRegression을 이용하여 학습/예측/평가 수행
lr_clf = LogisticRegression(solver='lbfgs', max_iter=100)
lr_clf.fit(X_train_tfidf_vect, y_train)
pred = lr_clf.predict(X_test_tfidf_vect)
print(f'TF-IDF Vectorized Logistic Regression의 예측 정확도는 {accuracy_score(y_test, pred):.3f}')
"""
TF-IDF Vectorized Logistic Regression의 예측 정확도는 0.692
"""

# GridSearchCV로 LogisticRegression C 하이퍼 파라미터 튜닝
from sklearn.model_selection import GridSearchCV, train_test_split

# 가벼운 튜닝: 학습 샘플 일부(3000) + C 후보/CV 축소 → 찾은 C로 전체 데이터 재학습
_, X_tune, _, y_tune = train_test_split(
    X_train_tfidf_vect, y_train, train_size=3000, stratify=y_train, random_state=156
)
lr_tune = LogisticRegression(solver='saga', max_iter=100, random_state=156)
params = {'C': [0.1, 1, 10]}
grid_cv_lr = GridSearchCV(lr_tune, param_grid=params, cv=2, scoring='accuracy')
grid_cv_lr.fit(X_tune, y_tune)
print('Logistic Regression best C parameter:', grid_cv_lr.best_params_)

lr_best = LogisticRegression(
    solver='saga', max_iter=100, C=grid_cv_lr.best_params_['C'], random_state=156
)
lr_best.fit(X_train_tfidf_vect, y_train)
pred = lr_best.predict(X_test_tfidf_vect)
print(f'TF-IDF Vectorized Logistic Regression(GridSearch 튜닝)의 예측 정확도는 {accuracy_score(y_test, pred):.3f}')
"""
TF-IDF Vectorized Logistic Regression(GridSearch 튜닝)의 예측 정확도는 0.702
"""