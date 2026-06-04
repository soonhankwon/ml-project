# Opinion Review 데이터 세트를 이용한 문서 군집화 수행하기
import pandas as pd
import glob, os

from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_colwidth', 700)

all_files = glob.glob(os.path.join('./topics', '*.data'))
filename_list = []
opinion_text = []

# 개별 파일들의 파일명은 filename_list 리스트로 취합
# 개별 파일들의 파일 내용은 DataFrame 로딩 후 다시 string으로 변환하여 opinion_text 리스트로 취합 
for file_ in all_files:
    # 개별 파일을 읽어서 DataFrame으로 생성 
    df = pd.read_table(file_, index_col=None, header=0, encoding='latin1')
    
    # 절대경로로 주어진 file 명을 가공. 만일 Linux에서 수행시에는 아래 \\를 / 변경. 
    # 맨 마지막 .data 확장자도 제거
    filename_ = file_.split('/')[-1]
    filename = filename_.split('.')[0]

    # 파일명 리스트와 파일 내용 리스트에 파일명과 파일 내용을 추가. 
    filename_list.append(filename)
    opinion_text.append(df.to_string())

# 파일명 리스트와 파일 내용 리스트를  DataFrame으로 생성
document_df = pd.DataFrame({'filename':filename_list, 'opinion_text':opinion_text})
print(document_df.head())

# TfidVectorizer의 tokenizer인자로 사용될 lemmatization 어근 변환 함수를 설정
from nltk.stem import WordNetLemmatizer
import nltk
import string

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmar = WordNetLemmatizer()

# 입력으로 들어온 token단어들에 대해서 lemmatization 어근 변환. 
def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]

# TfidfVectorizer 객체 생성 시 tokenizer인자로 해당 함수를 설정하여 lemmatization 적용
# 입력으로 문장을 받아서 stop words 제거-> 소문자 변환 -> 단어 토큰화 -> lemmatization 어근 변환. 
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# TF-IDF 기반 Vectorization 적용 및 KMeans 군집화 수행
tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', ngram_range=(1,2),
min_df=0.05, max_df=0.85)

# opinion_text 컬럼값으로 feature vectorization 수행
feature_vect = tfidf_vect.fit_transform(document_df['opinion_text'])

from sklearn.cluster import KMeans

# 5개 집합으로 군집화 수행. 예제를 위해 동일한 클러스터링 결과 도출용 random_state=0 
km_cluster = KMeans(n_clusters=5, max_iter=10000, random_state=0)
km_cluster.fit(feature_vect)
cluster_label = km_cluster.labels_
cluster_centers = km_cluster.cluster_centers_

document_df['cluster_label'] = cluster_label
print(document_df.head())
"""
                        filename  ... cluster_label
0     battery-life_ipod_nano_8gb  ...             1
1  gas_mileage_toyota_camry_2007  ...             4
2        room_holiday_inn_london  ...             3
3    location_holiday_inn_london  ...             0
4    staff_bestwestern_hotel_sfo  ...             3

[5 rows x 3 columns]
"""
