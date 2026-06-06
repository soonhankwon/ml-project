# 데이터 전처리

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

mercari_df = pd.read_csv(BASE_DIR / 'train.tsv', sep='\t')
print(mercari_df.shape)
print(mercari_df.head(3))
"""
(1482535, 8)
   train_id                                 name  ...  shipping                                   item_description
0         0  MLB Cincinnati Reds T Shirt Size XL  ...         1                                 No description yet
1         1     Razer BlackWidow Chroma Keyboard  ...         0  This keyboard is in great condition and works ...
2         2                       AVA-VIV Blouse  ...         1  Adorable top with a hint of lace and a key hol...

[3 rows x 8 columns]

shipping: 배송비 여부 - 1이면 무료, 0이면 유료
"""

# 타겟값의 분포도 확인
import matplotlib.pyplot as plt
import seaborn as sns

y_train_df = mercari_df['price']
plt.figure(figsize=(6,4))
sns.histplot(y_train_df, bins=100)
# plt.show()

import numpy as np
y_train_df = np.log1p(y_train_df)
sns.histplot(y_train_df, bins=50)
# plt.show()

mercari_df['price'] = np.log1p(mercari_df['price'])
print(mercari_df['price'].head(3))
"""
0    2.397895
1    3.970292
2    2.397895
Name: price, dtype: float64
"""

# 각 피처들의 유형 살펴보기
print('Shipping 값 유형:\n', mercari_df['shipping'].value_counts())
print('item_condition_id 값 유형:\n', mercari_df['item_condition_id'].value_counts())
"""
Shipping 값 유형:
 shipping
0    819435
1    663100
Name: count, dtype: int64

item_condition_id 값 유형:
 item_condition_id
1    640549
3    432161
2    375479
4     31962
5      2384
Name: count, dtype: int64
"""

boolean_cond = mercari_df['item_description'] == 'No description yet'
print(mercari_df[boolean_cond]['item_description'].count())
"""
82489
"""

# category name이 대/중/소와 같이 /문자열 기반으로 되어있음. 이를 개별 컬럼들로 재생성
def split_cat(category_name):
    try:
        return category_name.split('/')
    except:
        return ['Other_null', 'Other_null', 'Ohter_null']

def add_category_columns(df):
    df['cat_dae'], df['cat_jung'], df['cat_so'] = \
    zip(*df['category_name'].apply(lambda x: split_cat(x)))
    return df

# 위의 split_cat()을 apply lambda에서 호출하여 대, 중, 소 컬럼을 mercari_df에 생성
mercari_df = add_category_columns(mercari_df)

# 대분류만 값의 유형과 건수를 살펴보고, 중분류, 소분류는 값의 유형이 많으므로 분류 갯수만 추출
print('대분류 유형 :\n', mercari_df['cat_dae'].value_counts())
print('중분류 갯수 :', mercari_df['cat_jung'].nunique())
print('소분류 갯수 :', mercari_df['cat_so'].nunique())
"""
대분류 유형 :
 cat_dae
Women                     664385
Beauty                    207828
Kids                      171689
Electronics               122690
Men                        93680
Home                       67871
Vintage & Collectibles     46530
Other                      45351
Handmade                   30842
Sports & Outdoors          25342
Other_null                  6327
Name: count, dtype: int64
중분류 갯수 : 114
소분류 갯수 : 871
"""

def fill_missing_values(df):
    df['name'] = df['name'].fillna(value='Other_Null')
    df['brand_name'] = df['brand_name'].fillna(value='Other_Null')
    df['category_name'] = df['category_name'].fillna(value='Other_Null')
    df['item_description'] = df['item_description'].fillna(value='Other_Null')
    return df

mercari_df = fill_missing_values(mercari_df)

# 각 컬럼별로 Null값 건수 확인.
print(mercari_df.isnull().sum())
"""
train_id             0
name                 0
item_condition_id    0
category_name        0
brand_name           0
price                0
shipping             0
item_description     0
cat_dae              0
cat_jung             0
cat_so               0
dtype: int64
"""

# 피처 인코딩과 피처 벡터화
print('brand name 의 유형 건수 :', mercari_df['brand_name'].nunique())
print('brand name sample 5건 : \n', mercari_df['brand_name'].value_counts()[:5])
"""
brand name 의 유형 건수 : 4810
brand name sample 5건 : 
 brand_name
Other_Null           632682
PINK                  54088
Nike                  54043
Victoria's Secret     48036
LuLaRoe               31024
Name: count, dtype: int64
"""

print('name 의 종류 갯수 :', mercari_df['name'].nunique())
print('name sample 7건 : \n', mercari_df['name'][:7])
"""
name 의 종류 갯수 : 1225273
name sample 7건 : 
 0    MLB Cincinnati Reds T Shirt Size XL
1       Razer BlackWidow Chroma Keyboard
2                         AVA-VIV Blouse
3                  Leather Horse Statues
4                   24K GOLD plated rose
5       Bundled items requested for Ruie
6     Acacia pacific tides santorini top
Name: name, dtype: str
"""

pd.set_option('max_colwidth', 200)

# item_description의 평균 문자열 개수
print('item_description 평균 문자열 개수:', mercari_df['item_description'].str.len().mean())

mercari_df['item_description'][:2]
"""
item_description 평균 문자열 개수: 145.71139703278507
0                                                                                                                                                                              No description yet
1    This keyboard is in great condition and works like it came out of the box. All of the ports are tested and work perfectly. The lights are customizable via the Razer Synapse app on your PC.
Name: item_description, dtype: object
"""

# name은 Count로, item_description은 TF-IDF로 피처 벡터화
# name 속성에 대한 feature vectorization 변환
cnt_vec = CountVectorizer()
X_name = cnt_vec.fit_transform(mercari_df.name)

# item_description에 대한 feature vectorization 변환
tfidf_descp = TfidfVectorizer(max_features=50000, ngram_range=(1,3), stop_words='english')
X_descp = tfidf_descp.fit_transform(mercari_df['item_description'])

print('name vectorization shape:', X_name.shape)
print('item_description vectorization shape:', X_descp.shape)
"""
name vectorization shape: (1482535, 105757)
item_description vectorization shape: (1482535, 50000)
"""


# 원-핫 인코딩 변환 후 희소행렬 최적화 형태로 저장
from sklearn.preprocessing import OneHotEncoder
import numpy as np

oh_encoders = {}

def fit_onehot(column_name):
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    oh_encoders[column_name] = encoder
    return encoder.fit_transform(mercari_df[[column_name]])

# brand_name, item_condition_id, shipping 각 피처들을 희소 행렬 원-핫 인코딩 변환
X_brand = fit_onehot('brand_name')
X_item_cond_id = fit_onehot('item_condition_id')
X_shipping = fit_onehot('shipping')

# cat_dae, cat_jung, cat_so 각 피처들을 희소 행렬 원-핫 인코딩 변환
X_cat_dae = fit_onehot('cat_dae')
X_cat_jung = fit_onehot('cat_jung')
X_cat_so = fit_onehot('cat_so')

print(type(X_brand), type(X_item_cond_id), type(X_shipping))
print(f'X_brand_shape:{X_brand.shape}, X_item_cond_id shape:{X_item_cond_id.shape}')
print(f'X_shipping shape:{X_shipping.shape,}, X_cat_dae shape:{X_cat_dae.shape}')
print(f'X_cat_jung shape:{X_cat_jung.shape}, X_cat_so shape:{X_cat_so.shape}')
"""
<class 'scipy.sparse._csr.csr_matrix'> <class 'scipy.sparse._csr.csr_matrix'> <class 'scipy.sparse._csr.csr_matrix'>
X_brand_shape:(1482535, 4810), X_item_cond_id shape:(1482535, 5)
X_shipping shape:((1482535, 2),), X_cat_dae shape:(1482535, 11)
X_cat_jung shape:(1482535, 114), X_cat_so shape:(1482535, 871)
"""

from scipy.sparse import hstack
import gc

sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id, 
X_shipping, X_cat_dae, X_cat_jung, X_cat_so)

# scipy sparse 모듈의 hstack 함수를 이용하여 앞에서 인코딩과 Vectorization을 수행한 데이터셋을 모두 결합
X_features_sparse = hstack(sparse_matrix_list).tocsr()
print(type(X_features_sparse), X_features_sparse.shape)
"""
<class 'scipy.sparse._csr.csr_matrix'> (1482535, 161569)
"""

# 데이터 셋이 메모리를 많이 차지하므로 사용 용도가 끝났으면 바로 메모리에서 삭제. 
del X_features_sparse
gc.collect()

# 릿지 회귀 모델 구축 및 평가
def rmsle(y, y_pred):
    # underflow, overflow를 막기 위해 log가 아닌 log1p로 rmsle 계산
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_pred), 2)))

def evaluate_org_price(y_test, preds):
    # 원본 데이터는 log1p로 변환되었으므로 exmpm1으로 원복 필요
    preds_exmpm = np.expm1(preds)
    y_test_exmpm = np.expm1(y_test)

    # rmsle로 RMSLE값 추출
    rmsle_result = rmsle(y_test_exmpm, preds_exmpm)
    return rmsle_result

# 여러 모델에 대한 학습/예측을 수행하기 위해 별도의 함수인 model_train_predict() 생성
def model_train_predict(model, matrix_list):
    # scipy.sparse 모듈의 hstack을 이용하여 sparse matrix 결홥
    X = hstack(matrix_list).tocsr()

    X_train, X_test, y_train, y_test = train_test_split(X, mercari_df['price'], test_size=0.2, random_state=156)

    # 모델 학습 및 예측
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    del X, X_train, X_test, y_train
    gc.collect()

    return preds, y_test

# 릿지 선형 회귀로 학습/예측/평가. Item Desciprition 피처의 영향도를 알아보기 위한 테스트 함께 수행
linear_model = Ridge(solver='lsqr', fit_intercept=False)
sparse_matrix_list = (X_name, X_brand, X_item_cond_id, X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
linear_preds, y_test = model_train_predict(model=linear_model, matrix_list=sparse_matrix_list)
print('Item Description을 제외했을 때 rmsle 값:', evaluate_org_price(y_test , linear_preds))
"""
Item Description을 제외했을 때 rmsle 값: 0.49841531400517897
"""

sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id, X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
linear_preds, y_test = model_train_predict(model=linear_model, matrix_list=sparse_matrix_list)
print('Item Description을 포함한 rmsle 값:',  evaluate_org_price(y_test ,linear_preds))
"""
Item Description을 포함한 rmsle 값: 0.4681328386096162
"""

def transform_test_features(test_df):
    test_df = add_category_columns(test_df)
    test_df = fill_missing_values(test_df)

    X_test_name = cnt_vec.transform(test_df['name'])
    X_test_descp = tfidf_descp.transform(test_df['item_description'])
    X_test_brand = oh_encoders['brand_name'].transform(test_df[['brand_name']])
    X_test_item_cond_id = oh_encoders['item_condition_id'].transform(test_df[['item_condition_id']])
    X_test_shipping = oh_encoders['shipping'].transform(test_df[['shipping']])
    X_test_cat_dae = oh_encoders['cat_dae'].transform(test_df[['cat_dae']])
    X_test_cat_jung = oh_encoders['cat_jung'].transform(test_df[['cat_jung']])
    X_test_cat_so = oh_encoders['cat_so'].transform(test_df[['cat_so']])

    return hstack((
        X_test_descp,
        X_test_name,
        X_test_brand,
        X_test_item_cond_id,
        X_test_shipping,
        X_test_cat_dae,
        X_test_cat_jung,
        X_test_cat_so
    )).tocsr()

def make_ridge_submission(output_path='submission_ridge.csv'):
    train_features = hstack(sparse_matrix_list).tocsr()
    ridge_model = Ridge(solver='lsqr', fit_intercept=False)
    ridge_model.fit(train_features, mercari_df['price'])

    test_df = pd.read_csv(BASE_DIR / 'test.tsv', sep='\t')
    test_features = transform_test_features(test_df)
    predicted_log_prices = ridge_model.predict(test_features)
    predicted_prices = np.expm1(predicted_log_prices)
    predicted_prices = np.maximum(predicted_prices, 0)

    submission = pd.DataFrame({
        'test_id': test_df['test_id'],
        'price': predicted_prices
    })
    submission.to_csv(BASE_DIR / output_path, index=False)
    print(f'Ridge 제출 파일 생성 완료: {BASE_DIR / output_path}')

    del train_features, test_features
    gc.collect()

make_ridge_submission()

RUN_LIGHTGBM = False

if RUN_LIGHTGBM:
    # LightGBM 회귀모델 구축과 앙상블을 이용한 최종 예측 평가
    from lightgbm import LGBMRegressor

    lgbm_model =LGBMRegressor(n_estimator=200, learning_rate=0.5, num_leaves=125, random_state=156)
    lgbm_preds, y_test = model_train_predict(model=lgbm_model, matrix_list=sparse_matrix_list)
    print('LightGBM rmsle 값:',  evaluate_org_price(y_test, lgbm_preds))

    preds = lgbm_preds * 0.45 + linear_preds * 0.55
    print('LightGBM과 Ridge를 ensemble한 최종 rmsle 값:',  evaluate_org_price(y_test , preds))
    """
    LightGBM rmsle 값: 0.46521224321921983
    LightGBM과 Ridge를 ensemble한 최종 rmsle 값: 0.45061360013035756
    """

