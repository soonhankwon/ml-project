import numpy as np
# 단일 인덱싱
array1 = np.arange(start=1, stop=10)
print('array1:', array1)
value = array1[2] # 1D -> 0D
print('value:', value) # 3
print(type(value))

print('맨 뒤의 값:', array1[-1], ', 맨 뒤에서 두번째 값:', array1[-2])
"""
맨 뒤의 값: 9 , 맨 뒤에서 두번째 값: 8
"""

array1[0] = 9
array1[8] = 0
print('array1:', array1)
"""
array1: [9 2 3 4 5 6 7 8 0]
"""

array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)
print(array2d)

print('(row=0, col=0) Index 가리키는 값:', array2d[0,0]) # 1
print('(row=0, col=1) Index 가리키는 값:', array2d[0,1]) # 2
print('(row=1, col=0) Index 가리키는 값:', array2d[1,0]) # 4
print('(row=1, col=1) Index 가리키는 값:', array2d[1,1]) # 5

# 슬라이싱 인덱스
array1 = np.arange(start=1, stop=10)
print('array1:', array1)
array3 = array1[0:3]
print('array3:', array3)
print(type(array3))
"""
<class 'numpy.ndarray'>
"""

array1 = np.arange(start=1, stop=10)
## 위치 인덱스 0-2(2포함)까지 추출
array4 = array1[0:3]
print(array4)
## 위치 인덱스 3부터 마지막까지 추출
array5 = array1[3:]
print(array5)
## 위치 인덱스로 전체 데이터 추출
array6 = array1[:]
print(array6)

array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)
print('array2d:\n', array2d)

print('array2d[0:2, 0:2] \n', array2d[0:2, 0:2])
print('array2d[1:3, 0:3] \n', array2d[1:3, 0:3])
print('array2d[1:3, :] \n', array2d[1:3, :])
print('array2d[:, :] \n', array2d[:, :])
print('array2d[:2, 1:] \n', array2d[:2, 1:])
print('array2d[:2, 0] \n', array2d[:2, 0]) # 컬럼 단일인덱싱 2d -> 1d [1,4]
print(array2d[0])
print(array2d[1])

print('array2d[0] shape:', array2d[0].shape, 'array2d[1] shape:', array2d[1].shape) # (3,) (3,)

# 팬시 인덱싱
array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)

array3 = array2d[[0,1], 2]
print('array2d[[0,1], 2] => ', array3.tolist()) # [3,6]

array4 = array2d[[0,1], 0:2]
print('array2d[[0,1], 0:2] => ', array4.tolist()) # [[1,2], [4,5]]

# 불린 인덱싱
array3 = array1d[array1d > 5]
print('array1d > 5 boolean indexing 결과 값: ', array3)

val = array1d > 5
print(val, type(val), val.shape)
"""
[False False False False False  True  True  True  True] <class 'numpy.ndarray'> (9,)
"""
boolean_indexes = np.array([False, False, False, False, False, True, True, True, True])
array3 = array1d[boolean_indexes]
print('boolean indexing filtering 결과: ', array3) # boolean indexing filtering 결과:  [6 7 8 9]

target = []
# 불린 인덱싱을 적용하지 않았을 경우
for i in range(0, 9):
    if array1d[i] > 5:
        target.append(array1d[i])

array_selected = np.array(target)
print(array_selected) # [6 7 8 9]

# Fancy indexing
indexes = np.array([5,6,7,8])
print(array1d[indexes]) # [6 7 8 9]