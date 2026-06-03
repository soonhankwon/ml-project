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

# 개별 문서별 토픽 분포 확인
# LDA 객체의 transform()을 수행하면 개별 문서별 토픽 분포를 반환
doc_topics = lda.transform(feat_vect)
print(doc_topics.shape)
print(doc_topics[:3])

# 개별 문서별 토픽 분포도를 출력
# 20news group으로 만들어진 문서명을 출력
def get_filename_list(newsdata):
    filename_list = []

    for file in newsdata.filenames:
        filename_temp = file.split('\\')[-2:]
        filename = '.'.join(filename_temp)
        filename_list.append(filename)

    return filename_list

filename_list = get_filename_list(news_df)
print('filename 개수:', len(filename_list), 'filename list 10개만:', filename_list[:10])
"""
filename 개수: 7862 filename list 10개만: 
['/Users/soon/scikit_learn_data/20news_home/20news-bydate-train/soc.religion.christian/20630', 
'/Users/soon/scikit_learn_data/20news_home/20news-bydate-test/sci.med/59422', 
'/Users/soon/scikit_learn_data/20news_home/20news-bydate-test/comp.graphics/38765', 
'/Users/soon/scikit_learn_data/20news_home/20news-bydate-test/comp.graphics/38810', 
'/Users/soon/scikit_learn_data/20news_home/20news-bydate-test/sci.med/59449', 
'/Users/soon/scikit_learn_data/20news_home/20news-bydate-train/comp.graphics/38461', 
'/Users/soon/scikit_learn_data/20news_home/20news-bydate-train/comp.windows.x/66959', 
'/Users/soon/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104487', 
'/Users/soon/scikit_learn_data/20news_home/20news-bydate-train/sci.electronics/53875', 
'/Users/soon/scikit_learn_data/20news_home/20news-bydate-train/sci.electronics/53617']
"""

# DataFrame으로 생성하여 문서별 토픽 분포도 확인
import pandas as pd

topic_names = ['Topic #' + str(i) for i in range(0, 8)]
doc_topic_df = pd.DataFrame(data=doc_topics, columns=topic_names, index=filename_list)
print(doc_topic_df.head(10))
"""
                                                   Topic #0  Topic #1  Topic #2  ...  Topic #5  Topic #6  Topic #7
/Users/soon/scikit_learn_data/20news_home/20new...  0.013897  0.013944  0.013891  ...  0.013892  0.013935  0.434244
/Users/soon/scikit_learn_data/20news_home/20new...  0.277504  0.181518  0.002121  ...  0.002121  0.002121  0.002121
/Users/soon/scikit_learn_data/20news_home/20new...  0.005445  0.221666  0.005445  ...  0.005442  0.005442  0.745675
/Users/soon/scikit_learn_data/20news_home/20new...  0.005439  0.005441  0.005449  ...  0.388387  0.005442  0.005442
/Users/soon/scikit_learn_data/20news_home/20new...  0.006584  0.552000  0.006587  ...  0.006585  0.006588  0.006585
/Users/soon/scikit_learn_data/20news_home/20new...  0.008342  0.008352  0.182622  ...  0.008341  0.008343  0.008351
/Users/soon/scikit_learn_data/20news_home/20new...  0.372861  0.041667  0.377020  ...  0.041703  0.041667  0.041711
/Users/soon/scikit_learn_data/20news_home/20new...  0.225351  0.674669  0.004814  ...  0.004812  0.004812  0.004810
/Users/soon/scikit_learn_data/20news_home/20new...  0.008944  0.836686  0.008932  ...  0.109691  0.008932  0.008938
/Users/soon/scikit_learn_data/20news_home/20new...  0.041733  0.041720  0.708081  ...  0.041669  0.041699  0.041686
"""