from sklearn.datasets import load_digits
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class MyFakeClassifier(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

digits = load_digits()
y = (digits.target == 7).astype(int)
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=11)

fakeclf = MyFakeClassifier()
fakeclf.fit(X_train, y_train)

fake_predictions = fakeclf.predict(X_test)

# 앞절의 예측 결과인 fakepred와 실제 결과인 y_test의 Confusion Matrix를 출력
print(confusion_matrix(y_test, fake_predictions))
"""
[[405   0]
 [ 45   0]]
 TN: 405, FN: 45
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score

# MyFakeClassifier의 예측 결과로 정밀도와 재현율 측정
print('정밀도: ', precision_score(y_test, fake_predictions))
print('재현율: ', recall_score(y_test, fake_predictions))
"""
정밀도:  0.0
재현율:  0.0
"""

# 오차행렬, 정확도, 정밀도, 재현율을 한꺼번에 계산하는 함수 생성
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    print('오차행렬')
    print(confusion)
    print(f'정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}')
    
get_clf_eval(y_test, fake_predictions)
"""
[[405   0]
 [ 45   0]]
정확도: 0.9000, 정밀도: 0.0000, 재현율: 0.0000
"""


