import pandas as pd

train_df = pd.read_csv('ratings_train.txt', sep='\t')
print(train_df.head(3))
"""
         id                           document  label
0   9976970                아 더빙.. 진짜 짜증나네요 목소리      0
1   3819312  흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나      1
2  10265843                  너무재밓었다그래서보는것을추천한다      0
"""

print(train_df['label'].value_counts())
"""
label
0    75173
1    74827
Name: count, dtype: int64
"""

import re
train_df = train_df.fillna(' ')
# 정규 표현식을 이용하여 숫자를 공백으로 변경(정규 표현식으로 \d는 숫자를 의미함)
train_df['document'] = train_df['document'].apply(lambda x: re.sub(r"\d+", " ", x))

# 테스트 데이터 셋을 로딩하고 동일하게 Null 및 숫자를 공백으로 변환
test_df = pd.read_csv('ratings_test.txt', sep='\t')
test_df = test_df.fillna(' ')
test_df['document'] = test_df['document'].apply( lambda x : re.sub(r"\d+", " ", x) )

# id 칼럼 삭제 수행
train_df.drop('id', axis=1, inplace=True) 
test_df.drop('id', axis=1, inplace=True)

from konlpy.tag import Okt
import jpype
from pathlib import Path

def get_jvm_path():
    jvm_path = jpype.getDefaultJVMPath()
    if isinstance(jvm_path, bytes):
        jvm_path = jvm_path.decode()

    jvm_path = Path(jvm_path)
    if jvm_path.is_dir():
        jvm_path = jvm_path / 'lib/server/libjvm.dylib'

    return str(jvm_path)

okt = Okt(jvmpath=get_jvm_path())
def tw_tokenizer(text):
    # 입력 인자로 들어온 text를 형태소 단어로 토큰화하여 list 객체 반환
    tokens_ko = okt.morphs(text)
    return tokens_ko

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Okta 객체의 morphs() 이용한 tokenizer를 사용. ngram_range는 (1,2)
tfidf_vect = TfidfVectorizer(tokenizer=tw_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
tfidf_vect.fit(train_df['document'])
tfidf_matrix_train = tfidf_vect.transform(train_df['document'])

# Logistic Regression을 이용하여 감성 분석 Classification 수행
lg_clf = LogisticRegression(random_state=0, solver='liblinear')

# Parameter C 최적화를 위해 GridSearchCV를 이용
params = { 'C': [1 ,3.5, 4.5, 5.5, 10 ] }
grid_cv = GridSearchCV(lg_clf, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv.fit(tfidf_matrix_train, train_df['label'])
print(grid_cv.best_params_, round(grid_cv.best_score_, 4))
"""
Fitting 3 folds for each of 5 candidates, totalling 15 fits
{'C': 3.5} 0.8593
"""

from sklearn.metrics import accuracy_score

# 학습 데이터를 적용한 TfidVectorizer를 이용하여 테스트 데이터를 TF-IDF 값으로 Feature 변환함
tfidf_matrix_test = tfidf_vect.transform(test_df['document'])

# classifier는 GridSearchCV에서 최적 파라미터로 학습된 classifier를 그대로 이용
best_estimator = grid_cv.best_estimator_
preds = best_estimator.predict(tfidf_matrix_test)

print('Logistic Regression 정확도: ', accuracy_score(test_df['label'],preds))
"""
Logistic Regression 정확도:  0.86172
"""
