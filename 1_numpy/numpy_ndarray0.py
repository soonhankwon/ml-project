import numpy as np

list1 = [1,2,3]
print('list1 type:', type(list1))
# list1 type: <class 'list'>

array1 = np.array(list1)
print('array1 type', type(array1))
print('array1 array 형태', array1.shape)

"""
array1 type <class 'numpy.ndarray'>
array1 array 형태 (3,)
"""

array2 = np.array([[1,2,3], [2,3,4]])
print('array2 type', type(array2))
print('array2 array 형태', array2.shape)

"""
array2 type <class 'numpy.ndarray'>
array2 array 형태 (2, 3)
"""

array3 = np.array([[1,2,3]])
print('array3 type', type(array3))
print('array3 array 형태', array3.shape)

"""
array3 type <class 'numpy.ndarray'>
array3 array 형태 (1, 3)
"""

print('array1: {:0}차원, array2: {:1}차원, array3: {:2}차원'.format(array1.ndim, array2.ndim, array3.ndim))
"""
array1: 1차원, array2: 2차원, array3:  2차원
"""

print(type(array1))
print(array1, array1.dtype)
"""
<class 'numpy.ndarray'>
[1 2 3] int64
"""

list2 = [1,2,'test']
array2 = np.array(list2)
print(array2, array2.dtype)
"""
['1' '2' 'test'] <U21 (문자열로 자동형변환)
"""

list3 = [1,2,3.0]
array3 = np.array(list3)
print(array3, array3.dtype)
"""
[1. 2. 3.] float64 (큰 데이터 타입으로 자동형변환)
"""

array_int = np.array([1,2,3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)
"""
[1. 2. 3.] float64
"""

array_int1 = array_float.astype('int32')
print(array_int1, array_int1.dtype)
"""
[1 2 3] int32
"""

array_float1 = np.array([1.1, 2.1, 3.1])
array_int2 = array_float1.astype('int32')
print(array_int2, array_int2.dtype)
"""
[1 2 3] int32
"""

sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)
"""
[0 1 2 3 4 5 6 7 8 9]
int64 (10,)
"""