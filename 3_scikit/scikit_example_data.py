from sklearn.datasets import load_iris

iris_data = load_iris()
print(type(iris_data))
# <class 'sklearn.utils._bunch.Bunch'>

keys = iris_data.keys()
print('붓꽃 데이터 세트의 키들:', keys)
# 붓꽃 데이터 세트의 키들: dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

# 피처의 이름
print('\nfeature_name의 type:', type(iris_data.feature_names))
print('feature_names 의 shape:', len(iris_data.feature_names))
print(iris_data.feature_names)

"""
feature_name의 type: <class 'list'>
feature_names 의 shape: 4
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
"""

# 개별 레이블의 이름
print('\ntarget_name의 type:', type(iris_data.target_names))
print('target_names 의 shape:', len(iris_data.target_names))
print(iris_data.target_names)

"""
target_name의 type: <class 'numpy.ndarray'>
target_names 의 shape: 3
['setosa' 'versicolor' 'virginica']
"""

# 피처의 데이터 세트
print('\ndata의 type:', type(iris_data.data))
print('data 의 shape:', len(iris_data.data.shape))
print(iris_data.data)

"""
data의 type: <class 'numpy.ndarray'>
data 의 shape: 2
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 ...
"""

# 분류시 레이블값
print('\ntarget의 type:', type(iris_data.target))
print('target 의 shape:', len(iris_data.target.shape))
print(iris_data.target)

"""
target의 type: <class 'numpy.ndarray'>
target 의 shape: 1
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
"""