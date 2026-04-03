from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn import pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# random 값으로 구성된 X값에 대해 Consine 변환값을 반환
def true_fun(X):
    return np.cos(1.5 * np.pi * X)

#  X는 0부터 1까지 30개의 random값을 순서대로 sampling한 데이터
np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))

# y값은 cosine 기반의 true_fun()에서 약간의 Noise 변동값을 더한 값
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.scatter(X, y)
# plt.show()

plt.figure(figsize=(14, 5))
degrees = [1, 4, 15]

# 다항 회귀의 차수를 1, 4, 15로 각각 변화시키며 비교
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    # 개별 degree별로 Polynomial 변환
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([
        ('polynomial_features', polynomial_features), 
        ('linear_regression', linear_regression)
    ])
    pipeline.fit(X.reshape(-1,1), y)

    # 교차검증으로 다항회귀를 평가
    scores = cross_val_score(pipeline, X.reshape(-1,1), y, scoring="neg_mean_squared_error", cv=10)
    coefficients = pipeline.named_steps['linear_regression'].coef_
    print(f'\nDegree {degrees[i]}의 계수는 {np.round(coefficients, 2)}입니다.')
    print(f'Degree {degrees[i]} MSE는 {-1*np.mean(scores):.2f}입니다.')

    # 0부터 1까지 테스트 데이터 세트를 100개로 나눠 예측 수행
    # 테스트 데이터 세트에 회귀예측을 수행하고 예측 곡선과 실제 곡선을 비교
    X_test = np.linspace(0, 1, 100)
    # 예측값 곡선
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    # 실제값 곡선
    plt.plot(X_test, true_fun(X_test), '--', label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0,1))
    plt.ylim((-2,2))
    plt.legend(loc="best")
    plt.title(f"Degree {degrees[i]}\nMSE = {-scores.mean():.2e}(+/- {scores.std():.2e})")

plt.show()

"""
Degree 1의 계수는 [-1.61]입니다.
Degree 1 MSE는 0.41입니다.

Degree 4의 계수는 [  0.47 -17.79  23.59  -7.26]입니다.
Degree 4 MSE는 0.04입니다.

Degree 15의 계수는 [-2.98293000e+03  1.03899360e+05 -1.87416098e+06  2.03716227e+07
 -1.44873316e+08  7.09315656e+08 -2.47065940e+09  6.24561535e+09
 -1.15676620e+10  1.56895110e+10 -1.54006219e+10  1.06457414e+10
 -4.91378296e+09  1.35919876e+09 -1.70381099e+08]입니다.
Degree 15 MSE는 182621180.61입니다.
"""