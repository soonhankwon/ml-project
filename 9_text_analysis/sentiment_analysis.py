# 지도학습 기반 감성 분석 - IMDB 영화평

import pandas as pd
review_df = pd.read_csv('./word2vec-nlp-tutorial/labeledTrainData.tsv', header=0, sep="\t", quoting=3)
print(review_df.head(3))
"""
         id  sentiment                                             review
0  "5814_8"          1  "With all this stuff going down at the moment ...
1  "2381_9"          1  "\"The Classic War of the Worlds\" by Timothy ...
2  "7759_3"          0  "The film starts with a manager (Nicholas Bell...
"""

# 데이터 사전 처리 html태그 제거 및 숫자문자 제거
import re
# <br> html 태그는 replace 함수로 공백으로 변환
review_df['review'] = review_df['review'].str.replace('<br />', '')

# 파이썬의 정규 표현식 모듈인 re를 이용하여 영어 문자열이 아닌 문자는 모두 공백으로 변환
review_df['review'] = review_df['review'].apply(lambda x: re.sub("[^a-zA-Z]", " ", x))

# 학습/테스트 데이터 분리
from sklearn.model_selection import train_test_split

class_df = review_df['sentiment']
feature_df = review_df.drop(['id', 'sentiment'], axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(feature_df, class_df, test_size=0.3, random_state=156)
print(X_train.shape, X_test.shape)
"""
(17500, 1) (7500, 1)
"""

# Pipeline을 통해 Count기반 피처 벡터화 및 머신러닝 학습/예측/평가
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# 스톱 워드는 English, filtering, ngram은 (1,2)로 설정해 CountVectorization 수행
# LogisticRegression의 C는 10으로 설정
pipeline = Pipeline([
    ('cnt_vect', CountVectorizer(stop_words='english', ngram_range=(1,2) )),
    ('lr_clf', LogisticRegression(solver='liblinear', C=10))])

# Pipeline 객체를 이용하여 fit(), predict()로 학습/예측 수행, predict_proba()는 roc_auc때문에 수행
pipeline.fit(X_train['review'], y_train)
pred = pipeline.predict(X_test['review'])
pred_probs = pipeline.predict_proba(X_test['review'])[:, 1]
print(f'예측 정확도는: {accuracy_score(y_test, pred)}, ROC-AUC는: {roc_auc_score(y_test, pred_probs)}')
"""
예측 정확도는: 0.8861333333333333, ROC-AUC는: 0.9502294132711131
"""

# Pipeline을 통해 TF-IDF 기반 피처 벡터화 및 머신러닝 학습/예측/평가
# 스톱 워드는 english, filtering, ngram은 (1,2)로 설정해 TF-IDF 벡터화 수행. 
# LogisticRegression의 C는 10으로 설정. 
pipeline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1,2) )),
    ('lr_clf', LogisticRegression(solver='liblinear', C=10))])

pipeline.fit(X_train['review'], y_train)
pred = pipeline.predict(X_test['review'])
pred_probs = pipeline.predict_proba(X_test['review'])[:,1]

print(f'예측 정확도는: {accuracy_score(y_test, pred)}, ROC-AUC는: {roc_auc_score(y_test, pred_probs)}')
"""
예측 정확도는: 0.8938666666666667, ROC-AUC는: 0.9598077196676531
"""

# TF-IDF 기반 캐글 제출 파일 생성 (전체 학습 데이터로 재학습)
DATA_DIR = './word2vec-nlp-tutorial'

test_df = pd.read_csv(f'{DATA_DIR}/testData.tsv', header=0, sep='\t', quoting=3)
test_df['id'] = test_df['id'].str.strip('"')
test_df['review'] = test_df['review'].str.replace('<br />', '')
test_df['review'] = test_df['review'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

submission_pipeline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('lr_clf', LogisticRegression(solver='liblinear', C=10)),
])
submission_pipeline.fit(review_df['review'], review_df['sentiment'])

submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': submission_pipeline.predict(test_df['review']),
})
submission_path = f'{DATA_DIR}/tfidf_submission.csv'
submission.to_csv(submission_path, index=False)
print(f'캐글 제출 파일 생성 완료: {submission_path} ({len(submission)}건)')