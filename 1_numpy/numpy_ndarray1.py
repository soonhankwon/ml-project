import array
import numpy as np

# ndarray 편하게 생성하기 - arange, zeros, ones
sequence_array = np.arange(10)
print(sequence_array) # [0 1 2 3 4 5 6 7 8 9]
print(sequence_array.dtype, sequence_array.shape) # int64 (10,)

zero_array = np.zeros((3,2), dtype='int32')
print(zero_array)
print(zero_array, zero_array.shape)
"""
[[0 0]
 [0 0]
 [0 0]]
[[0 0]
 [0 0]
 [0 0]] (3, 2)
"""
one_array = np.ones((3,2), dtype='int32')
print(one_array)
print(one_array, one_array.shape)
"""
[[1 1]
 [1 1]
 [1 1]]
[[1 1]
 [1 1]
 [1 1]] (3, 2)
"""

# ndarray 차원과 크기를 변경하는 reshape
array1 = np.arange(10)
print('array1:\n', array1)
"""
array1: 
[0 1 2 3 4 5 6 7 8 9]
"""

# (2,5) shape로 변환
array2 = array1.reshape(2, 5)
print('array2:\n', array2)
"""
array2:
[[0 1 2 3 4][5 6 7 8 9]]
"""

# (5,2) shape로 변환
array3 = array2.reshape(5, 2)
print('array3:\n', array3)
"""
array3: 
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
"""

array2 = array1.reshape(-1, 5)
print('array2 shape', array2.shape) # array2 shape (2, 5)

array3 = array1.reshape(5, -1)
print('array3 shape', array3.shape) # array3 shape (5, 2)

array1 = np.arange(8)
array3d = array1.reshape([2,2,2]) # 1차원 ndarray를 3차원 ndarray로 변환
print('array3d:\n', array3d.tolist())
"""
array3d:
 [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
"""

# 3차원 ndarray를 2차원 ndarray로 변환하되 컬럼갯수는 1
array5 = array3d.reshape(-1, 1)
print('array5:\n', array5.tolist())
print('array5 shape', array5.shape)
"""
array5:
 [[0], [1], [2], [3], [4], [5], [6], [7]]
array5 shape (8, 1)
"""

# 1차원 ndarray를 2차원 ndarray로 변환하되 컬럼갯수는 1
array6 = array1.reshape(-1, 1)
print('array6:\n', array6.tolist())
print('array6 shape', array6.shape)
"""
array6:
 [[0], [1], [2], [3], [4], [5], [6], [7]]
array6 shape (8, 1)
"""

# 3차원 array를 1차원으로 변환
array1d = array3d.reshape(-1,)
print(array1d) # [0 1 2 3 4 5 6 7]
