# 20 Newsgroup 토픽 모델링
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 20개 중 8개의 주제 데이터 로드 및 Count기반 피처 벡터화. LDA는 Count기반 Vectorizer만 적용
# 모터사이클, 야구, 그래픽스, 윈도우즈, 중동, 기독교, 의학, 우주 주제를 추출. 
cats = ['rec.motorcycles', 'rec.sport.baseball', 'comp.graphics', 'comp.windows.x',
        'talk.politics.mideast', 'soc.religion.christian', 'sci.electronics', 'sci.med']

# 위에서 cats 변수로 기재된 category만 추출. fetch_20newsgroups()의 categories에 cats 입력
news_df = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), 
categories=cats, random_state=0)

# LDA는 Count 기반의 Vectorizer만 적용
count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words='english', ngram_range=(1,2))
feat_vect = count_vect.fit_transform(news_df.data)
print('CountVectorizer Shape:', feat_vect.shape)
"""
CountVectorizer Shape: (7862, 1000)
"""

# LDA 객체 생성후 Count 피처 벡터화 객체로 LDA 수행
lda = LatentDirichletAllocation(n_components=8, random_state=0)
lda.fit(feat_vect)

# 각 토픽 모델링 주제별 단어들의 연관도 확인
# LDA 객체의 componets_ 속성은 주제별로 개별 단어들의 연관도 정규화 숫자가 들어있음
print(lda.components_.shape)
print(lda.components_)
"""
(8, 1000) 주제 개수, 피처단어 개수
숫자가 클수록 토픽에서 단어가 차지하는 비중이 높음
[[3.60992018e+01 1.35626798e+02 2.15751867e+01 ... 3.02911688e+01
  8.66830093e+01 6.79285199e+01]
 [1.25199920e-01 1.44401815e+01 1.25045596e-01 ... 1.81506995e+02
  1.25097844e-01 9.39593286e+01]
 [3.34762663e+02 1.25176265e-01 1.46743299e+02 ... 1.25105772e-01
  3.63689741e+01 1.25025218e-01]
 ...
 [3.60204965e+01 2.08640688e+01 4.29606813e+00 ... 1.45056650e+01
  8.33854413e+00 1.55690009e+01]
 [1.25128711e-01 1.25247756e-01 1.25005143e-01 ... 9.17278769e+01
  1.25177668e-01 3.74575887e+01]
 [5.49258690e+01 4.47009532e+00 9.88524814e+00 ... 4.87048440e+01
  1.25034678e-01 1.25074632e-01]]
"""

# 각 토픽별 중심 단어 확인
def display_topics(model, feature_names, no_top_words):
    for topic_index, topic in enumerate(model.components_):
        print('Topic #',topic_index)

        # components_ array에서 가장 값이 큰 순으로 정렬했을 때, 그 값의 array index를 반환. 
        topic_word_indexes = topic.argsort()[::-1]
        top_indexes=topic_word_indexes[:no_top_words]
        
        # top_indexes대상인 index별로 feature_names에 해당하는 word feature 추출 후 join으로 concat
        feature_concat = ' '.join([feature_names[i] for i in top_indexes])                
        print(feature_concat)

# CountVectorizer 객체내의 전체 word들의 명칭을 get_features_names()를 통해 추출
feature_names = count_vect.get_feature_names_out()

# Topic별 가장 연관도가 높은 word를 15개만 추출
display_topics(lda, feature_names, 15)
"""
Topic # 0
year 10 game medical health team 12 20 disease cancer 1993 games years patients good
Topic # 1
don just like know people said think time ve didn right going say ll way
Topic # 2
image file jpeg program gif images output format files color entry 00 use bit 03
Topic # 3
like know don think use does just good time book read information people used post
Topic # 4
armenian israel armenians jews turkish people israeli jewish government war dos dos turkey arab armenia 000
Topic # 5
edu com available graphics ftp data pub motif mail widget software mit information version sun
Topic # 6
god people jesus church believe christ does christian say think christians bible faith sin life
Topic # 7
use dos thanks windows using window does display help like problem server need know run
"""

