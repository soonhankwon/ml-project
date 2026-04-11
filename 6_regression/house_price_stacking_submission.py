"""
Kaggle House Prices — 스태킹 회귀(Lasso 메타)로 submission CSV 생성.
house_price_pred.py 와 동일한 전처리·베이스/메타 하이퍼파라미터를 사용합니다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

BASE = Path(__file__).resolve().parent
DATA = BASE / "house-prices"


def preprocess_train_test(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """train/test 동일 파이프라인으로 정렬된 피처 행렬과 (로그) 타깃 반환."""
    y = np.log1p(train_raw["SalePrice"].astype(float))

    drop_cols = ["Id", "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]
    train_x = train_raw.drop(columns=["SalePrice"] + drop_cols, errors="ignore")
    test_x = test_raw.drop(columns=drop_cols, errors="ignore")

    num_means = train_x.mean(numeric_only=True)
    train_x = train_x.fillna(num_means)
    test_x = test_x.fillna(num_means)

    for col in train_x.select_dtypes(include=["object"]).columns:
        mode = train_x[col].mode()
        fill = mode.iloc[0] if len(mode) else ""
        train_x[col] = train_x[col].fillna(fill)
        test_x[col] = test_x[col].fillna(fill)

    num_cols = train_x.select_dtypes(include=[np.number]).columns
    skew_features = train_x[num_cols].apply(lambda x: skew(x))
    skew_top = skew_features[skew_features > 1].index
    train_x[skew_top] = np.log1p(train_x[skew_top])
    test_x[skew_top] = np.log1p(test_x[skew_top])

    n_train = len(train_x)
    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    ohe = pd.get_dummies(combined)
    ohe.columns = ohe.columns.str.replace(" ", "_")

    tr = ohe.iloc[:n_train].copy()
    te = ohe.iloc[n_train:].copy()
    cond1 = tr["GrLivArea"] > np.log1p(4000)
    cond2 = y.values < np.log1p(500000)
    out = cond1.values & cond2
    tr = tr.loc[~out].reset_index(drop=True)
    y = pd.Series(y.values[~out])

    return tr, y, te.reset_index(drop=True)


def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=False)
    train_fold_pred = np.zeros((X_train_n.shape[0], 1))
    test_pred = np.zeros((X_test_n.shape[0], n_folds))
    print(model.__class__.__name__, " model 시작 ")

    for folder_counter, (train_index, valid_index) in enumerate(kf.split(X_train_n)):
        print("\t 폴드 세트: ", folder_counter, " 시작 ")
        X_tr = X_train_n[train_index]
        y_tr = y_train_n[train_index]
        X_te = X_train_n[valid_index]

        model.fit(X_tr, y_tr)
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1, 1)
        test_pred[:, folder_counter] = model.predict(X_test_n)

    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)
    return train_fold_pred, test_pred_mean


def main():
    train_raw = pd.read_csv(DATA / "train.csv")
    test_raw = pd.read_csv(DATA / "test.csv")
    test_ids = test_raw["Id"].copy()

    X_tr, y_tr, X_te = preprocess_train_test(train_raw, test_raw)

    X_train_n = X_tr.values
    y_train_n = y_tr.values
    X_test_n = X_te.values

    ridge_reg = Ridge(alpha=8)
    lasso_reg = Lasso(alpha=0.001)
    xgb_reg = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        colsample_bytree=0.5,
        subsample=0.8,
        random_state=156,
        n_jobs=-1,
    )
    lgbm_reg = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=4,
        subsample=0.6,
        colsample_bytree=0.4,
        reg_lambda=10,
        n_jobs=-1,
        force_row_wise=True,
        random_state=156,
    )

    ridge_train, ridge_test = get_stacking_base_datasets(
        ridge_reg, X_train_n, y_train_n, X_test_n, 5
    )
    lasso_train, lasso_test = get_stacking_base_datasets(
        lasso_reg, X_train_n, y_train_n, X_test_n, 5
    )
    xgb_train, xgb_test = get_stacking_base_datasets(
        xgb_reg, X_train_n, y_train_n, X_test_n, 5
    )
    lgbm_train, lgbm_test = get_stacking_base_datasets(
        lgbm_reg, X_train_n, y_train_n, X_test_n, 5
    )

    stack_train = np.concatenate(
        (ridge_train, lasso_train, xgb_train, lgbm_train), axis=1
    )
    stack_test = np.concatenate(
        (ridge_test, lasso_test, xgb_test, lgbm_test), axis=1
    )

    meta = Lasso(alpha=0.0005)
    meta.fit(stack_train, y_train_n)
    pred_log = meta.predict(stack_test)
    pred_sale = np.expm1(pred_log)
    pred_sale = np.clip(pred_sale, 1.0, None)

    out = pd.DataFrame({"Id": test_ids, "SalePrice": pred_sale})
    submit_path = DATA / "submission_stacking.csv"
    out.to_csv(submit_path, index=False)
    print(f"저장 완료: {submit_path}")

    # 홀드아웃 없이 재현용: 학습 데이터에 대한 OOF 메타 예측 RMSE (참고)
    oof_meta = meta.predict(stack_train)
    rmse = np.sqrt(mean_squared_error(y_train_n, oof_meta))
    print("스태킹 메타 모델 OOF RMSE (로그 스케일, 참고):", round(rmse, 6))


if __name__ == "__main__":
    main()
