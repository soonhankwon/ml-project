import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

cust_df = pd.read_csv("./santander-customer-satisfaction/train.csv", encoding='latin-1')

# var3 피처값 대체 및 ID 피처 드롭
cust_df['var3'] = cust_df['var3'].replace(-999999, 2)
cust_df.drop('ID', axis=1, inplace=True)

# 피처 세트와 레이블 세트분리. 레이블 컬럼은 DataaFrame의 맨 마지막에 위치해 컬럼 위치 -1로 분리
X_features = cust_df.iloc[:, :-1] # 맨마지막 컬럼 제외
y_labels = cust_df.iloc[:, -1] # 맨마지막 컬럼만
print(f'피처 데이터 shape:{X_features.shape}')
"""
피처 데이터 shape:(76020, 369)
"""

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=0, stratify=y_labels)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

from lightgbm import LGBMClassifier, early_stopping

lgbm_clf = LGBMClassifier(n_estimators=500)
eval_set = [(X_tr, y_tr), (X_val, y_val)]

lgbm_clf.fit(
    X_tr, y_tr,
    eval_set=eval_set,
    eval_metric="auc",
    callbacks=[early_stopping(stopping_rounds=50, verbose=True)],
)

lgbm_roc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:, 1])
print(f'ROC AUC: {lgbm_roc_score}')
"""
ROC AUC: 0.8185898509329037
"""

from hyperopt import hp
from sklearn.model_selection import KFold

lgbm_search_space = {'num_leaves': hp.quniform('num_leaves', 32, 64, 1),
                     'max_depth': hp.quniform('max_depth', 100, 160, 1),
                     'min_child_samples': hp.quniform('min_child_samples', 60, 100, 1),
                     'subsample': hp.uniform('subsample', 0.7, 1),
                     'learning_rate': hp.uniform('learning_rate', 0.01, 0.2)}

def objective_func(search_space):
    lgbm_clf =  LGBMClassifier(n_estimators=100, num_leaves=int(search_space['num_leaves']),
                               max_depth=int(search_space['max_depth']),
                               min_child_samples=int(search_space['min_child_samples']), 
                               subsample=search_space['subsample'],
                               learning_rate=search_space['learning_rate'])
    # 3개 k-fold 방식으로 평가된 roc_auc 지표를 담는 list
    roc_auc_list = []
    
    # 3개 k-fold방식 적용 
    kf = KFold(n_splits=3)
    # X_train을 다시 학습과 검증용 데이터로 분리
    for tr_index, val_index in kf.split(X_train):
        # kf.split(X_train)으로 추출된 학습과 검증 index값으로 학습과 검증 데이터 세트 분리 
        X_tr, y_tr = X_train.iloc[tr_index], y_train.iloc[tr_index]
        X_val, y_val = X_train.iloc[val_index], y_train.iloc[val_index]

        # early stopping은 30회로 설정하고 추출된 학습과 검증 데이터로 XGBClassifier 학습 수행. 
        lgbm_clf.fit(X_tr, y_tr, eval_metric="auc",
           eval_set=[(X_tr, y_tr), (X_val, y_val)], 
           callbacks=[early_stopping(stopping_rounds=30, verbose=True)])

        # 1로 예측한 확률값 추출후 roc auc 계산하고 평균 roc auc 계산을 위해 list에 결과값 담음.
        score = roc_auc_score(y_val, lgbm_clf.predict_proba(X_val)[:, 1]) 
        roc_auc_list.append(score)
    
    # 3개 k-fold로 계산된 roc_auc값의 평균값을 반환하되, 
    # HyperOpt는 목적함수의 최소값을 위한 입력값을 찾으므로 -1을 곱한 뒤 반환.
    return -1 * np.mean(roc_auc_list)

from hyperopt import fmin, tpe, Trials
trials = Trials()

# fmin()함수를 호출. max_evals지정된 횟수만큼 반복 후 목적함수의 최소값을 가지는 최적 입력값 추출. 
best = fmin(fn=objective_func, space=lgbm_search_space, algo=tpe.suggest,
            max_evals=50, # 최대 반복 횟수를 지정합니다.
            trials=trials)

print('best:', best)

lgbm_clf =  LGBMClassifier(n_estimators=500, num_leaves=int(best['num_leaves']),
                           max_depth=int(best['max_depth']),
                           min_child_samples=int(best['min_child_samples']), 
                           subsample=round(best['subsample'], 5),
                           learning_rate=round(best['learning_rate'], 5))


# evaluation metric을 auc로, early stopping은 100 으로 설정하고 학습 수행. 
lgbm_clf.fit(
    X_tr, y_tr, 
    eval_metric="auc", 
    eval_set=[(X_tr, y_tr), (X_val, y_val)], 
    callbacks=[early_stopping(stopping_rounds=100, verbose=True)])

lgbm_roc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:,1])
print(f'ROC AUC: {lgbm_roc_score:.4f}')
"""
ROC AUC: 0.8197
"""

# 캐글 제출용 CSV 생성
test_df = pd.read_csv("./santander-customer-satisfaction/test.csv", encoding='latin-1')
test_ids = test_df['ID'].copy()
test_df['var3'] = test_df['var3'].replace(-999999, 2)
test_df.drop('ID', axis=1, inplace=True)
X_test_sub = test_df
pred_proba = lgbm_clf.predict_proba(X_test_sub)[:, 1]
submission = pd.DataFrame({'ID': test_ids, 'TARGET': pred_proba})
submission.to_csv("./santander-customer-satisfaction/submission2.csv", index=False)
print(f'제출 파일 생성 완료: submission2.csv ({len(submission)} rows)')