# KDE(Kernel Density Estimation)의 이해
# seaborn의 displot()을 이용하여 KDE 시각화
import numpy as np
from scipy.linalg import bandwidth
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)

np.random.seed(0)
x = np.random.normal(0, 1, size=30)
print(x)
"""
[ 1.76405235  0.40015721  0.97873798  2.2408932   1.86755799 -0.97727788
  0.95008842 -0.15135721 -0.10321885  0.4105985   0.14404357  1.45427351
  0.76103773  0.12167502  0.44386323  0.33367433  1.49407907 -0.20515826
  0.3130677  -0.85409574 -2.55298982  0.6536186   0.8644362  -0.74216502
  2.26975462 -1.45436567  0.04575852 -0.18718385  1.53277921  1.46935877]
"""
sns.distplot(x)
plt.show()
sns.distplot(x, rug=True)
plt.show()
sns.distplot(x, kde=False, rug=True)
plt.show()
sns.distplot(x, hist=False, rug=True)
plt.show()

# 개별 관측데이터에 대해 가우시안 커널함수를 적용
from scipy import stats

bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.) # 최적 bandwith 근사
support = np.linspace(-4, 4, 200)

kernels= []
for x_i in x:
    kernel = stats.norm(x_i, bandwidth).pdf(support)
    kernels.append(kernel)
    plt.plot(support, kernel, color='r')

sns.rugplot(x, color=".2", linewidth=3)
plt.show()

from scipy.integrate import trapezoid
density = np.sum(kernels, axis=0)
density /= trapezoid(density, support)
plt.plot(support, density)
plt.show()

# seaborn은 kdeplot()으로 kde곡선을 바로 구할 수 있음
sns.kdeplot(x, shade=True)

# bandwidth에 따른 KDE 변화
sns.kdeplot(x)
sns.kdeplot(x, bw=0.2, label="bw: 0.2")
sns.kdeplot(x, bw=2, label="bw: 2")
plt.legend()
plt.show()

# 사이킷런을 이용한 Mean Shift
# make_blobs()을 이용하여 2개의 feature와 3개의 군집 중심점을 가지는 임의의 데이터 200개를 생성하고 MeanShift를 이용하여 군집화 수행
X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.7, random_state=0)

meanshift = MeanShift(bandwidth=0.8)
cluster_labels = meanshift.fit_predict(X)
print('cluster labels 유형', np.unique(cluster_labels))

# 커널함수의 bandwidth 크기를 1로 약간 증가 후에 MeanShift 군집화 재 수행
meanshift = MeanShift(bandwidth=1)
cluster_labels = meanshift.fit_predict(X)
print('cluster labels 유형', np.unique(cluster_labels))
"""
cluster labels 유형 [0 1 2 3 4 5]
cluster labels 유형 [0 1 2]
"""

# 최적의 bandwidth값을 estimate_bandwidth()로 계산한 뒤에 다시 군집화 수행
from sklearn.cluster import estimate_bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.25)
print('bandwidth 값:', round(bandwidth, 3))
"""
bandwidth 값: 1.518
"""

import pandas as pd

clusterDF = pd.DataFrame(data=X, columns=['ftr1', 'ftr2'])
clusterDF['target'] = y

# estimate_banwidth()로 최적의 bandwidth 계산
best_bandwidth = estimate_bandwidth(X)

meanshift = MeanShift(bandwidth=best_bandwidth)
cluster_labels = meanshift.fit_predict(X)
print('cluster labels 유형:', np.unique(cluster_labels))
"""
cluster labels 유형: [0 1 2]
"""

clusterDF['meanshift_label'] = cluster_labels
centers = meanshift.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ['o', 's', '^', 'x', '*']

for label in unique_labels:
    label_cluster = clusterDF[clusterDF['meanshift_label'] == label]
    center_x_y = centers[label]
    # 군집별로 다른 마커로 산점도 적용
    plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], edgecolor='k', marker=markers[label])
    
    # 군집별 중심 표현
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color='gray', alpha=0.9, marker=markers[label])
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', edgecolor='k', marker='$%d$' % label)

plt.show()
print(clusterDF.groupby('target')['meanshift_label'].value_counts())
"""
target  meanshift_label
0       0                  67
1       1                  67
2       2                  66
Name: count, dtype: int64
"""

