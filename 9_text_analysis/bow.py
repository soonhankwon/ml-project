# Bag of Words - BOW
# 사이킷런 CountVectorizer 테스트
text_sample_01 = 'The Matrix is everywhere its all around us, here even in this room. \
You can see it out your window or on your television\
You feel it when you go to work, or go to church or pay your taxes'
text_sample_02 = 'You take the blue pill and the story ends. You wake in your bed and you belive whatever you want to believe\
You take the red pill and you stau in Wonderland and I show you how deep the rabbit-hole goes'
text = []
text.append(text_sample_01)
text.append(text_sample_02)
print(text, '\n', len(text))
"""
['The Matrix is everywhere its all around us, here even in this room. You can see it out your window or on your televisionYou feel it when you go to work, or go to church or pay your taxes', 'You take the blue pill and the story ends. You wake in your bed and you belive whatever you want to believeYou take the red pill and you stau in Wonderland and I show you how deep the rabbit-hole goes'] 
 2
"""

# CountVectorizer 객체 생성 후 fit(), transform()으로 텍스트에 대한 feature vectorization 수행
from sklearn.feature_extraction.text import CountVectorizer

# Count Vectorization으로 feature extraction 변환 수행
cnt_vect = CountVectorizer()
cnt_vect.fit(text)

ftr_vect = cnt_vect.transform(text)

# 피처 벡터화 후 데이터 유형 및 여러 속성 확인
print(type(ftr_vect), ftr_vect.shape)
print(ftr_vect)
"""
<class 'scipy.sparse._csr.csr_matrix'> (2, 52)
<Compressed Sparse Row sparse matrix of dtype 'int64'
        with 57 stored elements and shape (2, 52)>
  Coords        Values
  (0, 0)        1
  (0, 2)        1
  (0, 7)        1
  (0, 8)        1
  (0, 11)       1
  (0, 12)       1
  (0, 13)       1
  (0, 14)       2
  (0, 16)       1
  (0, 19)       1
  (0, 20)       1
  (0, 21)       2
  (0, 22)       1
  (0, 23)       1
  (0, 24)       1
  (0, 25)       3
  (0, 26)       1
  (0, 27)       1
  (0, 31)       1
  (0, 32)       1
  (0, 37)       1
  (0, 38)       1
  (0, 39)       1
  (0, 40)       1
  (0, 41)       2
  :     :
  (1, 3)        1
  (1, 4)        1
  (1, 5)        1
  (1, 6)        1
  (1, 9)        1
  (1, 10)       1
  (1, 15)       1
  (1, 17)       1
  (1, 18)       1
  (1, 19)       2
  (1, 28)       2
  (1, 29)       1
  (1, 30)       1
  (1, 33)       1
  (1, 34)       1
  (1, 35)       1
  (1, 36)       2
  (1, 39)       4
  (1, 41)       1
  (1, 43)       1
  (1, 44)       1
  (1, 45)       1
  (1, 48)       1
  (1, 50)       6
  (1, 51)       1
"""
print(cnt_vect.vocabulary_)
"""
{'the': 39, 'matrix': 23, 'is': 20, 'everywhere': 12, 'its': 22, 'all': 0, 'around': 2, 'us': 42, 'here': 16, 'even': 11, 'in': 19, 'this': 40, 'room': 31, 'you': 50, 'can': 7, 'see': 32, 'it': 21, 'out': 26, 'your': 51, 'window': 47, 'or': 25, 'on': 24, 'televisionyou': 38, 'feel': 13, 'when': 46, 'go': 14, 'to': 41, 'work': 49, 'church': 8, 'pay': 27, 'taxes': 37, 'take': 36, 'blue': 6, 'pill': 28, 'and': 1, 'story': 35, 'ends': 10, 'wake': 43, 'bed': 3, 'belive': 5, 'whatever': 45, 'want': 44, 'believeyou': 4, 'red': 30, 'stau': 34, 'wonderland': 48, 'show': 33, 'how': 18, 'deep': 9, 'rabbit': 29, 'hole': 17, 'goes': 15}
"""

cnt_vect = CountVectorizer(max_features=5, stop_words='english')
cnt_vect.fit(text)
ftr_vect = cnt_vect.transform(text)
print(type(ftr_vect), ftr_vect.shape)
print(cnt_vect.vocabulary_)
"""
<class 'scipy.sparse._csr.csr_matrix'> (2, 5)
{'window': np.int64(4), 'televisionyou': np.int64(1), 'pill': np.int64(0), 'wake': np.int64(2), 'want': np.int64(3)}
"""

# ngram_range 확인
cnt_vect = CountVectorizer(ngram_range=(1, 2))
cnt_vect.fit(text)
ftr_vect = cnt_vect.transform(text)
print(type(ftr_vect), ftr_vect.shape)
print(cnt_vect.vocabulary_)
"""
<class 'scipy.sparse._csr.csr_matrix'> (2, 125)
{'the': 83, 'matrix': 50, 'is': 43, 'everywhere': 26, 'its': 48, 'all': 0, 'around': 6, 'us': 95, 'here': 33, 'even': 24, 'in': 39, 'this': 89, 'room': 68, 'you': 111, 'can': 16, 'see': 70, 'it': 45, 'out': 58, 'your': 120, 'window': 105, 'or': 54, 'on': 52, 'televisionyou': 81, 'feel': 28, 'when': 103, 'go': 30, 'to': 91, 'work': 109, 'church': 18, 'pay': 60, 'taxes': 80, 'the matrix': 85, 'matrix is': 51, 'is everywhere': 44, 'everywhere its': 27, 'its all': 49, 'all around': 1, 'around us': 7, 'us here': 96, 'here even': 34, 'even in': 25, 'in this': 40, 'this room': 90, 'room you': 69, 'you can': 113, 'can see': 17, 'see it': 71, 'it out': 46, 'out your': 59, 'your window': 124, 'window or': 106, 'or on': 56, 'on your': 53, 'your televisionyou': 123, 'televisionyou feel': 82, 'feel it': 29, 'it when': 47, 'when you': 104, 'you go': 114, 'go to': 31, 'to work': 94, 'work or': 110, 'or go': 55, 'to church': 93, 'church or': 19, 'or pay': 57, 'pay your': 61, 'your taxes': 122, 'take': 78, 'blue': 14, 'pill': 62, 'and': 2, 'story': 76, 'ends': 22, 'wake': 97, 'bed': 8, 'belive': 12, 'whatever': 101, 'want': 99, 'believeyou': 10, 'red': 66, 'stau': 74, 'wonderland': 107, 'show': 72, 'how': 37, 'deep': 20, 'rabbit': 64, 'hole': 35, 'goes': 32, 'you take': 117, 'take the': 79, 'the blue': 84, 'blue pill': 15, 'pill and': 63, 'and the': 4, 'the story': 88, 'story ends': 77, 'ends you': 23, 'you wake': 118, 'wake in': 98, 'in your': 42, 'your bed': 121, 'bed and': 9, 'and you': 5, 'you belive': 112, 'belive whatever': 13, 'whatever you': 102, 'you want': 119, 'want to': 100, 'to believeyou': 92, 'believeyou take': 11, 'the red': 87, 'red pill': 67, 'you stau': 116, 'stau in': 75, 'in wonderland': 41, 'wonderland and': 108, 'and show': 3, 'show you': 73, 'you how': 115, 'howdeep': 38, 'deep the': 21, 'the rabbit': 86, 'rabbit hole': 65, 'hole goes': 36}
"""
