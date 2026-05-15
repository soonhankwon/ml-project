# 붓꽃데이터 셋을 이용한 클러스터 평가
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 실루엣 분석 metric 값을 구하기 위한 API 추가
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
irisDF = pd.DataFrame(data=iris.data, columns=feature_names)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0).fit(irisDF)

irisDF['cluster'] = kmeans.labels_

# iris의 모든 개별 데이터에 실루엣 계수를 구함
score_samples = silhouette_samples(iris.data, irisDF['cluster'])
print('silhouette_samples() return 값의 shape', score_samples.shape)
"""
silhouette_samples() return 값의 shape (150,)
"""

# irisDF에 실루엣 계수 컬럼 추가
irisDF['silhouette_coeff'] = score_samples

# 모든 데이터의 평균 실루엣 계수값을 구함
average_score = silhouette_score(iris.data, irisDF['cluster'])
print(f'붓꽃 데이터 셋 Silhouette Analysis Score:{average_score:.3f}')

print(irisDF.head(3))
irisDF.groupby('cluster')['silhouette_coeff'].mean()
"""
붓꽃 데이터 셋 Silhouette Analysis Score:0.551
   sepal_length  sepal_width  ...  cluster  silhouette_coeff
0           5.1          3.5  ...        1          0.852582
1           4.9          3.0  ...        1          0.814916
2           4.7          3.2  ...        1          0.828797

[3 rows x 6 columns]
"""

# 클러스터별 평균 실루엣 계수의 시각화를 통한 클러스터 개수 최적화 방법
def visualize_silhouette(cluster_lists, X_features):
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math

    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)

    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 subfigures를 가지는 axs 생성
    fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)

    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화


