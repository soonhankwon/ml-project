import numpy as np

A = np.array(
    [[1,2,3], 
    [4,5,6]]
    )
B = np.array(
    [[7,8],
    [9,10],
    [11,12]]
    )

dot_product = np.dot(A,B)
print('행렬 내적 결과: \n', dot_product)
"""
행렬 내적 결과: 
 [[ 58  64]
 [139 154]]
"""

transpose_mat = np.transpose(B)
print('B의 전치행렬: \n', transpose_mat)
"""
B의 전치행렬: 
 [[ 7  9 11]
 [ 8 10 12]]
"""