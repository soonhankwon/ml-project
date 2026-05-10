from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()
print('target name:', iris.target_names)
irisDF = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
print(irisDF.head(3))
"""
target name: ['setosa' 'versicolor' 'virginica']
   sepal_length  sepal_width  petal_length  petal_width
0           5.1          3.5           1.4          0.2
1           4.9          3.0           1.4          0.2
2           4.7          3.2           1.3          0.2
"""

# KMeans 객체를 생성하고 군집화 수행
# labels_ 속성을 통해 각 데이터 포인트별로 할당된 군집 중심점(Centroid) 확인
# fit_predict(), fit_transform() 수행결과 확인

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(irisDF)
print(kmeans.labels_)
"""
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 2 2 2 2 0 2 2 2 2
 2 2 0 0 2 2 2 2 0 2 0 2 0 2 2 0 0 2 2 2 2 2 0 2 2 2 2 0 2 2 2 0 2 2 2 0 2
 2 0]
"""
kmeans.fit_predict(irisDF)
kmeans.fit_transform(irisDF)

# 군집화 결과를 irisDF에 cluster 컬럼으로 추가하고 target 값과 결과 비교
irisDF['target'] = iris.target
irisDF['cluster'] = kmeans.labels_
iris_result = irisDF.groupby(['target', 'cluster'])['sepal_length'].count()
print(iris_result)
"""
target  cluster
0       1          50
1       0          47
        2           3
2       0          14
        2          36
"""

# 2차원 평면에 데이터 포인트별로 군집화된 결과를 나타내기 위해 2차원 PCA값으로 각 데이터 차원축소
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca_transformed = pca.fit_transform(iris.data)

irisDF['pca_x'] = pca_transformed[:, 0]
irisDF['pca_y'] = pca_transformed[:, 1]
print(irisDF.head(3))
"""
   sepal_length  sepal_width  petal_length  petal_width  target  cluster     pca_x     pca_y
0           5.1          3.5           1.4          0.2       0        1 -2.684126  0.319397
1           4.9          3.0           1.4          0.2       0        1 -2.714142 -0.177001
2           4.7          3.2           1.3          0.2       0        1 -2.888991 -0.144949
"""