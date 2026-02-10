import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# 1. 데이터 로드 (Train + Test 합쳐서 전처리)
# One-Hot Encoding 시 컬럼 개수가 달라지는 문제를 방지
train_df = pd.read_csv('./spaceship_titanic_dataset/train.csv')
test_df = pd.read_csv('./spaceship_titanic_dataset/test.csv')

# 전처리를 위해 잠시 합침 (구분을 위해 플래그 추가)
train_df['dataset_type'] = 'train'
test_df['dataset_type'] = 'test'
all_data = pd.concat([train_df, test_df]).reset_index(drop=True)

# 2. 전처리 함수 정의
def preprocessing(df):
    # (1) 결측치 처리
    # 금액 관련: 0으로 채움 (대부분 소비 안함)
    billing_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[billing_features] = df[billing_features].fillna(0)

    # 범주형: 최빈값
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 나이: 중앙값
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Cabin: 결측치 임시 처리
    df['Cabin'] = df['Cabin'].fillna('N/-1/N')

    # (2) Feature Engineering
    # Cabin 분해: Deck(구역), Num(방번호), Side(좌/우)
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Num'] = df['Num'].astype(int) # 숫자로 변환

    # 총 소비액 & 소비 여부
    df['TotalSpend'] = df[billing_features].sum(axis=1)
    df['NoSpending'] = (df['TotalSpend'] == 0).astype(int)

    # 그룹 크기 (PassengerId 활용: gggg_pp)
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    group_size_map = df['Group'].value_counts()
    df['GroupSize'] = df['Group'].map(group_size_map)

    # (3) 불필요 컬럼 제거
    # Group, Cabin은 분해했으므로 원본 제거
    df = df.drop(['PassengerId', 'Name', 'Cabin', 'Group'], axis=1)

    # (4) 인코딩 (One-Hot Encoding)
    dummy_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    # drop_first=True로 다중공선성 문제 완화
    df = pd.get_dummies(df, columns=dummy_cols, drop_first=True, dtype=int)

    return df

# 전처리 수행
all_data_processed = preprocessing(all_data)

# 데이터 다시 분리
train_df_proc = all_data_processed[all_data_processed['dataset_type'] == 'train']
test_df_proc = all_data_processed[all_data_processed['dataset_type'] == 'test']

# 학습 데이터 준비
# Transported는 bool 타입이므로 int로 변환
y_train_full = train_df['Transported'].astype(int)
X_train_full = train_df_proc.drop(['Transported', 'dataset_type'], axis=1)
X_test_submit = test_df_proc.drop(['Transported', 'dataset_type'], axis=1)

# 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=11)

# 3. 모델 학습 및 평가
def get_clf_eval(y_test, pred, title):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred)
    print(f'=== [{title}] ===')
    print('오차행렬:\n', confusion)
    print(f'정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}, ROC: {roc_auc:.4f}\n')

# (1) 기본 모델 학습
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)

dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_val)
get_clf_eval(y_val, dt_pred, 'Decision Tree (Basic)')

rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_val)
get_clf_eval(y_val, rf_pred, 'Random Forest (Basic)')

# (2) RandomForest 하이퍼파라미터 튜닝 (GridSearchCV)
params = {
    'n_estimators': [100, 200],  # 트리 개수
    'max_depth': [8, 10, 12],    # 트리 깊이 제한 (너무 깊으면 과적합)
    'min_samples_split': [2, 5], # 분할 기준
    'min_samples_leaf': [1, 4]   # 리프 노드 최소 샘플 수
}

# n_jobs=-1: 모든 CPU 코어 사용
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=5, n_jobs=-1, scoring='accuracy')
grid_cv.fit(X_train, y_train)

print('GridSearchCV 최적 파라미터:', grid_cv.best_params_)
print(f'GridSearchCV 최고 정확도 (Train CV): {grid_cv.best_score_:.4f}')

# 최적 모델로 검증 데이터 재평가
best_rf = grid_cv.best_estimator_
best_pred = best_rf.predict(X_val)
get_clf_eval(y_val, best_pred, 'Random Forest (Tuned)')

# (3) 피처 중요도 시각화
ftr_importances_values = best_rf.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(10, 8))
plt.title('Feature Importances Top 20')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()

# (4) 제출 파일 생성
best_rf.fit(X_train_full, y_train_full)
test_pred = best_rf.predict(X_test_submit)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': test_pred.astype(bool) # 원본 형식이 bool이므로 변환
})

submission.to_csv('submission.csv', index=False)
print("submission.csv 저장 완료")

"""
=== [Decision Tree (Basic)] ===
오차행렬:
 [[613 207]
 [235 684]]
정확도: 0.7458, 정밀도: 0.7677, 재현율: 0.7443, F1: 0.7558, ROC: 0.7459

=== [Random Forest (Basic)] ===
오차행렬:
 [[690 130]
 [213 706]]
정확도: 0.8028, 정밀도: 0.8445, 재현율: 0.7682, F1: 0.8046, ROC: 0.8048

GridSearchCV 최적 파라미터: {'max_depth': 12, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}
GridSearchCV 최고 정확도 (Train CV): 0.8059
=== [Random Forest (Tuned)] ===
오차행렬:
 [[665 155]
 [181 738]]
정확도: 0.8068, 정밀도: 0.8264, 재현율: 0.8030, F1: 0.8146, ROC: 0.8070
"""