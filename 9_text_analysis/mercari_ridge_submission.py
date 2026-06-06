from pathlib import Path
import gc
import subprocess

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


KAGGLE_INPUT_DIR = Path('/kaggle/input/mercari-price-suggestion-challenge')
KAGGLE_OUTPUT_DIR = Path('/kaggle/working')
LOCAL_DIR = Path(__file__).resolve().parent

INPUT_DIR = KAGGLE_INPUT_DIR if KAGGLE_INPUT_DIR.exists() else LOCAL_DIR
OUTPUT_DIR = KAGGLE_OUTPUT_DIR if KAGGLE_OUTPUT_DIR.exists() else LOCAL_DIR


def ensure_tsv_file(file_name):
    tsv_path = INPUT_DIR / file_name
    if tsv_path.exists():
        return tsv_path

    working_tsv_path = OUTPUT_DIR / file_name
    if working_tsv_path.exists():
        return working_tsv_path

    compressed_path = INPUT_DIR / f'{file_name}.7z'
    if compressed_path.exists():
        subprocess.run(['7z', 'x', str(compressed_path), f'-o{OUTPUT_DIR}'], check=True)
        return working_tsv_path

    raise FileNotFoundError(f'{file_name} 또는 {file_name}.7z 파일을 찾을 수 없습니다.')


def split_cat(category_name):
    if not isinstance(category_name, str):
        return ['Other_null', 'Other_null', 'Other_null']

    categories = category_name.split('/')
    categories = categories[:3]
    categories += ['Other_null'] * (3 - len(categories))
    return categories


def preprocess(df):
    df = df.copy()
    df['name'] = df['name'].fillna('Other_Null')
    df['brand_name'] = df['brand_name'].fillna('Other_Null')
    df['category_name'] = df['category_name'].fillna('Other_Null')
    df['item_description'] = df['item_description'].fillna('Other_Null')

    df['cat_dae'], df['cat_jung'], df['cat_so'] = zip(
        *df['category_name'].apply(split_cat)
    )
    return df


def fit_onehot(train_df, column_name, encoders):
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    encoders[column_name] = encoder
    return encoder.fit_transform(train_df[[column_name]])


def build_train_features(train_df):
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        stop_words='english'
    )
    onehot_encoders = {}

    x_name = count_vectorizer.fit_transform(train_df['name'])
    x_descp = tfidf_vectorizer.fit_transform(train_df['item_description'])
    x_brand = fit_onehot(train_df, 'brand_name', onehot_encoders)
    x_item_cond_id = fit_onehot(train_df, 'item_condition_id', onehot_encoders)
    x_shipping = fit_onehot(train_df, 'shipping', onehot_encoders)
    x_cat_dae = fit_onehot(train_df, 'cat_dae', onehot_encoders)
    x_cat_jung = fit_onehot(train_df, 'cat_jung', onehot_encoders)
    x_cat_so = fit_onehot(train_df, 'cat_so', onehot_encoders)

    train_features = hstack((
        x_descp,
        x_name,
        x_brand,
        x_item_cond_id,
        x_shipping,
        x_cat_dae,
        x_cat_jung,
        x_cat_so
    )).tocsr()

    transformers = {
        'count_vectorizer': count_vectorizer,
        'tfidf_vectorizer': tfidf_vectorizer,
        'onehot_encoders': onehot_encoders
    }
    return train_features, transformers


def build_test_features(test_df, transformers):
    count_vectorizer = transformers['count_vectorizer']
    tfidf_vectorizer = transformers['tfidf_vectorizer']
    onehot_encoders = transformers['onehot_encoders']

    x_name = count_vectorizer.transform(test_df['name'])
    x_descp = tfidf_vectorizer.transform(test_df['item_description'])
    x_brand = onehot_encoders['brand_name'].transform(test_df[['brand_name']])
    x_item_cond_id = onehot_encoders['item_condition_id'].transform(test_df[['item_condition_id']])
    x_shipping = onehot_encoders['shipping'].transform(test_df[['shipping']])
    x_cat_dae = onehot_encoders['cat_dae'].transform(test_df[['cat_dae']])
    x_cat_jung = onehot_encoders['cat_jung'].transform(test_df[['cat_jung']])
    x_cat_so = onehot_encoders['cat_so'].transform(test_df[['cat_so']])

    return hstack((
        x_descp,
        x_name,
        x_brand,
        x_item_cond_id,
        x_shipping,
        x_cat_dae,
        x_cat_jung,
        x_cat_so
    )).tocsr()


def main():
    train_path = ensure_tsv_file('train.tsv')
    test_path = ensure_tsv_file('test.tsv')

    train_df = preprocess(pd.read_csv(train_path, sep='\t'))
    test_df = preprocess(pd.read_csv(test_path, sep='\t'))

    y_train = np.log1p(train_df['price'])
    train_features, transformers = build_train_features(train_df)

    model = Ridge(solver='lsqr', fit_intercept=False)
    model.fit(train_features, y_train)

    test_features = build_test_features(test_df, transformers)
    predicted_log_prices = model.predict(test_features)
    predicted_prices = np.maximum(np.expm1(predicted_log_prices), 0)

    submission = pd.DataFrame({
        'test_id': test_df['test_id'],
        'price': predicted_prices
    })
    output_path = OUTPUT_DIR / 'submission_ridge.csv'
    submission.to_csv(output_path, index=False)

    print(f'Saved submission: {output_path}')
    print(submission.head())

    del train_features, test_features
    gc.collect()


if __name__ == '__main__':
    main()
