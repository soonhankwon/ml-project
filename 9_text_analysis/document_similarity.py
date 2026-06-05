# 문서 유도 측정 방법 - 코사인 유사도
import sklearn
import numpy as np

def cos_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    l2_norm = (np.sqrt(sum(np.square(v1)))) * np.sqrt(sum(np.square(v2)))
    similarity = dot_product / l2_norm
    return similarity

from sklearn.feature_extraction.text import TfidfVectorizer

doc_list = ['if you take the blue pill, the story ends' ,
            'if you take the red pill, you stay in Wonderland',
            'if you take the red pill, I show you how deep the rabbit hole goes']

tfidf_vect_simple = TfidfVectorizer()
feature_vect_simple = tfidf_vect_simple.fit_transform(doc_list)
print(feature_vect_simple)
"""
<Compressed Sparse Row sparse matrix of dtype 'float64'
        with 29 stored elements and shape (3, 18)>
  Coords        Values
  (0, 6)        0.24543855687841593
  (0, 17)       0.24543855687841593
  (0, 14)       0.24543855687841593
  (0, 15)       0.49087711375683185
  (0, 0)        0.41556360057939173
  (0, 8)        0.24543855687841593
  (0, 13)       0.41556360057939173
  (0, 2)        0.41556360057939173
  (1, 6)        0.2340286519091622
  (1, 17)       0.4680573038183244
  (1, 14)       0.2340286519091622
  (1, 15)       0.2340286519091622
  (1, 8)        0.2340286519091622
  (1, 10)       0.3013544995034864
  (1, 12)       0.39624495215024286
  (1, 7)        0.39624495215024286
  (1, 16)       0.39624495215024286
  (2, 6)        0.1830059506093466
  (2, 17)       0.3660119012186932
  (2, 14)       0.1830059506093466
  (2, 15)       0.3660119012186932
  (2, 8)        0.1830059506093466
  (2, 10)       0.23565348175165166
  (2, 11)       0.3098560092999078
  (2, 5)        0.3098560092999078
  (2, 1)        0.3098560092999078
  (2, 9)        0.3098560092999078
  (2, 4)        0.3098560092999078
  (2, 3)        0.3098560092999078
"""

# TFidVectorizer로 transform()한 결과는 Sparse Matrix이므로 Dense Matrix로 변환
feature_vect_dense = feature_vect_simple.todense()

# 첫번쨰 문장과 두번째 문장의 feature vector 추출
vect1 = np.array(feature_vect_dense[0]).reshape(-1,)
vect2 = np.array(feature_vect_dense[1]).reshape(-1,)

# 첫번째 문장과 두번쨰 문장의 feature vector로 두개 문장의 Cosine 유사도 추출
similarity_simple = cos_similarity(vect1, vect2)
print(f'문장 1, 문장 2 Cosine 유사도: {similarity_simple:.3f}')
"""
문장 1, 문장 2 Cosine 유사도: 0.402
"""

vect1 = np.array(feature_vect_dense[0]).reshape(-1,)
vect3 = np.array(feature_vect_dense[2]).reshape(-1,)
similarity_simple = cos_similarity(vect1, vect3 )
print(f'문장 1, 문장 3 Cosine 유사도: {similarity_simple:.3f}')
"""
문장 1, 문장 3 Cosine 유사도: 0.404
"""

vect2 = np.array(feature_vect_dense[1]).reshape(-1,)
vect3 = np.array(feature_vect_dense[2]).reshape(-1,)
similarity_simple = cos_similarity(vect2, vect3 )
print(f'문장 2, 문장 3 Cosine 유사도: {similarity_simple:.3f}')
"""
문장 2, 문장 3 Cosine 유사도: 0.456
"""

from sklearn.metrics.pairwise import cosine_similarity

similarity_simple_pair = cosine_similarity(feature_vect_simple[0], feature_vect_simple)
print(similarity_simple_pair)
"""
[[1.         0.40207758 0.40425045]]
"""

similarity_simple_pair = cosine_similarity(feature_vect_simple[0] , feature_vect_simple[1:])
print(similarity_simple_pair)
"""
[[0.40207758 0.40425045]]
"""

similarity_simple_pair = cosine_similarity(feature_vect_simple , feature_vect_simple)
print(similarity_simple_pair)
print('shape:', similarity_simple_pair.shape)
"""
[[1.         0.40207758 0.40425045]
 [0.40207758 1.         0.45647296]
 [0.40425045 0.45647296 1.        ]]
 shape: (3, 3)
"""

