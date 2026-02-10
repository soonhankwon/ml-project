import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

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
    # 0. 그룹(PassengerId 활용: gggg_pp)
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])

    # HomePlanet: 같은 그룹에 있는 사람의 HomePlanet을 가져옴
    group_homeplanet = df.groupby('Group')['HomePlanet'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    df.loc[df['HomePlanet'].isna(), 'HomePlanet'] = df['Group'].map(group_homeplanet)
    
    # 그래도 비어있는 값(그룹원 전체가 NaN인 경우)은 전체 최빈값으로 채움
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])

    group_size_map = df['Group'].value_counts()
    df['GroupSize'] = df['Group'].map(group_size_map)

    # 1. 결측치 처리
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

    # 2. Feature Engineering
    # Cabin 분해: Deck(구역), Num(방번호), Side(좌/우)
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Num'] = df['Num'].astype(int) # 숫자로 변환

    # 총 소비액 & 소비 여부
    df['TotalSpend'] = df[billing_features].sum(axis=1)
    df['NoSpending'] = (df['TotalSpend'] == 0).astype(int)

    # 3. 불필요 컬럼 제거
    # Group, Cabin은 분해했으므로 원본 제거
    df = df.drop(['PassengerId', 'Name', 'Cabin', 'Group'], axis=1)

    # 4. 인코딩 (One-Hot Encoding)
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

# 3. 모델 학습 (3 앙상블)
# XGBoost (하이퍼파라미터 튜닝됨)
xgb_clf = XGBClassifier(
    n_estimators=1000,     # 넉넉하게 잡고 early_stopping으로 조절
    learning_rate=0.05,    # 학습률을 낮춰서 꼼꼼하게 학습
    max_depth=6,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    n_jobs=-1,
    random_state=11,
    enable_categorical=True # 범주형 데이터 처리 최적화
)

# LightGBM (속도 빠름 & 고성능)
lgbm_clf = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,         # 트리의 복잡도 조절 (max_depth보다 중요)
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=11,
    n_jobs=-1,
    verbose=-1
)

# Random Forest (기존 최적 모델 - 안정성 담당)
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=4,
    min_samples_split=2,
    random_state=11,
    n_jobs=-1
)

# --- 보팅(Voting) 분류기 생성 ---
# 서로 다른 3개의 모델이 투표를 해서 결과를 도출
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_clf),
        ('lgbm', lgbm_clf),
        ('rf', rf_clf)
    ],
    voting='soft'
)

# VotingClassifier 학습
voting_clf.fit(X_train, y_train)

# 검증
pred = voting_clf.predict(X_val)
accuracy = accuracy_score(y_val, pred)
print(f'\n[Voting Ensemble] 검증 세트 정확도: {accuracy:.4f}')

# --- 최종 제출 파일 생성 ---
# 전체 데이터로 재학습
voting_clf.fit(X_train_full, y_train_full)
test_pred = voting_clf.predict(X_test_submit)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': test_pred.astype(bool)
})

submission.to_csv('submission_ensemble.csv', index=False)