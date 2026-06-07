# 컨텐츠 기반 필터링 - TMDB 5000 Movie Dataset
import pandas as pd
import numpy as np

movies = pd.read_csv('./tmdb_5000_movies.csv')
print(movies.shape)
print(movies.head(1))
"""
(4803, 20)
      budget                                             genres  ... vote_average  vote_count
0  237000000  [{"id": 28, "name": "Action"}, {"id": 12, "nam...  ...          7.2       11800

[1 rows x 20 columns]
"""

movies_df = movies[['id', 'title', 'genres', 'vote_average', 'vote_count', 
'popularity', 'keywords', 'overview']]

pd.set_option('max_colwidth', 100)
movies_df[['genres', 'keywords']][:1]

# 텍스트 문자 1차 가공. 파이썬 딕셔너리 변환 후 리스트 형태로 변환
from ast import literal_eval

movies_df['genres'] = movies_df['genres'].apply(literal_eval)
movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)
movies_df['genres'] = movies_df['genres'].apply(lambda x: [y['name'] for y in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [y['name'] for y in x])
print(movies_df[['genres', 'keywords']][:1])
"""
                                          genres                      keywords
0  [Action, Adventure, Fantasy, Science Fiction]  [culture clash, future, space war, space colony, society, space travel, futuristic, romance, spa...
"""

# 장르 콘텐츠 필터링을 이용한 영화 추천. 장르 문자열을 Count 벡터화 후에 코사인 유사도로 각 영화를 비교
# 장르 문자열의 Count 기반 벡처 벡터화
from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer를 적용하기 위해 공백문자로 word 단위가 구분되는 문자열로 변환
movies_df['genres_literal'] = movies_df['genres'].apply(lambda x: (' ').join(x))
count_vect = CountVectorizer(min_df=0.0, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])
print(genre_mat.shape)
"""
(4803, 276)
"""

# 장르에 따른 영화별 코사인 유사도 추출
from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)
print(genre_sim.shape)
print(genre_sim[:2])
"""
(4803, 4803)
[[1.         0.59628479 0.4472136  ... 0.         0.         0.        ]
 [0.59628479 1.         0.4        ... 0.         0.         0.        ]]
"""

genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
print(genre_sim_sorted_ind[:1])
"""
[[   0 3494  813 ... 3038 3037 2401]]
"""

# 특정 영화와 장르별 유사도가 높은 영화를 반환하는 함수 생성
def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    
    # 인자로 입력된 movies_df DataFrame에서 'title' 컬럼이 입력된 title_name 값인 DataFrame추출
    title_movie = df[df['title'] == title_name]
    
    # title_named을 가진 DataFrame의 index 객체를 ndarray로 반환하고 
    # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n 개의 index 추출
    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]
    
    # 추출된 top_n index들 출력. top_n index는 2차원 데이터 임. 
    #dataframe에서 index로 사용하기 위해서 1차원 array로 변경
    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)
    
    return df.iloc[similar_indexes]

similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, 'The Godfather', 10)
print(similar_movies[['title', 'vote_average']])
"""
[[1370 4041 3337 1847 3378 4217 2839  281  588 3866]]
                                title  vote_average
1370                               21           6.5
4041                  This Is England           7.4
3337                    The Godfather           8.4
1847                       GoodFellas           8.2
3378                       Auto Focus           6.1
4217                             Kids           6.8
2839                         Rounders           6.9
281                 American Gangster           7.4
588   Wall Street: Money Never Sleeps           5.8
3866                      City of God           8.1
"""

# 평점이 높은 영화 정보 확인
print(movies_df[['title', 'vote_average', 'vote_count']].sort_values('vote_average', ascending=False)[:10])
"""
                         title  vote_average  vote_count
3519          Stiff Upper Lips          10.0           1
4247     Me You and Five Bucks          10.0           2
4045     Dancer, Texas Pop. 81          10.0           1
4662            Little Big Top          10.0           1
3992                 Sardaarji           9.5           2
2386            One Man's Hero           9.3           2
2970        There Goes My Baby           8.5           2
1881  The Shawshank Redemption           8.5        8205
2796     The Prisoner of Zenda           8.4          11
3337             The Godfather           8.4        5893
"""

# 평가 횟수에 대한 가중치가 부여된 평점(Weighted Rating) 계산
# 가중 평점(Weighted Rating) = (v/(v+m))R + (m/(v+m))C
"""
v: 개별 영화에 평점을 투표한 횟수
m: 평점을 부여하기 위한 최소 투표 횟수
R: 개별 영화에 대한 평균 평점
C: 전체 영화에 대한 평균 평점
"""
C = movies_df['vote_average'].mean()
m = movies_df['vote_count'].quantile(0.6)
print('C:', round(C, 3), 'm:', round(m,3))

percentile = 0.6
m = movies_df['vote_count'].quantile(percentile)
C = movies_df['vote_average'].mean()

def weighted_vote_average(record):
    v = record['vote_count']
    R = record['vote_average']
    
    return ( (v/(v+m)) * R ) + ( (m/(m+v)) * C )

movies_df['weighted_vote'] = movies_df.apply(weighted_vote_average, axis=1)
print(movies_df[['title', 'vote_average', 'weighted_vote', 'vote_count']].sort_values('weighted_vote', ascending=False)[:10])
"""
                         title  vote_average  weighted_vote  vote_count
1881  The Shawshank Redemption           8.5       8.396052        8205
3337             The Godfather           8.4       8.263591        5893
662                 Fight Club           8.3       8.216455        9413
3232              Pulp Fiction           8.3       8.207102        8428
65             The Dark Knight           8.2       8.136930       12002
1818          Schindler's List           8.3       8.126069        4329
3865                  Whiplash           8.3       8.123248        4254
809               Forrest Gump           8.2       8.105954        7927
2294             Spirited Away           8.3       8.105867        3840
2731    The Godfather: Part II           8.3       8.079586        3338
"""

def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    title_movie = df[df['title'] == title_name]
    title_index = title_movie.index.values

    # top_n의 2배에 해당하는 장르 유사성이 높은 index 추출
    similar_indexes = sorted_ind[title_index, :(top_n * 2)]
    similar_indexes = similar_indexes.reshape(-1)

    # 기준 영화 index는 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]

    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]

similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, 'The Godfather',10)
print(similar_movies[['title', 'vote_average', 'weighted_vote']])
"""
                            title  vote_average  weighted_vote
1881     The Shawshank Redemption           8.5       8.396052
2731       The Godfather: Part II           8.3       8.079586
1847                   GoodFellas           8.2       7.976937
3866                  City of God           8.1       7.759693
1663  Once Upon a Time in America           8.2       7.657811
892                        Casino           7.8       7.423040
281             American Gangster           7.4       7.141396
4041              This Is England           7.4       6.739664
1149              American Hustle           6.8       6.717525
1243                 Mean Streets           7.2       6.626569
"""