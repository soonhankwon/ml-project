import hyperopt

print(hyperopt.__version__) # 0.2.7

from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, Trials
import numpy as np

# -10 ~ 10까지 1간격을 가지는 입력 변수 x와 -15 ~ 15까지 1간격으로 입력 변수 y설정
search_space = {'x': hp.quniform('x', -10, 10, 1), 'y': hp.quniform('y', -15, 15, 1)}

# 목적 함수를 생성. 변숫값과 변수 검색 공간을 가지는 딕셔너리를 인자로 받고, 특정 값을 반환
def objective_fun(search_space):
    x = search_space['x']
    y = search_space['y']
    retval = x**2 - 20*y
    return retval

# 입력 결괏값을 저장한 Trials 객체값 생성
trial_val = Trials()

# 목적 함수의 최솟값을 반환하는 최적 입력 변숫값을 5번의 입력값 시도(max_evals=5)로 찾아냄
best_01 = fmin(fn=objective_fun, space=search_space, algo=tpe.suggest, max_evals=5, trials=trial_val, rstate=np.random.default_rng(seed=0))

print('best:', best_01)
"""
best: {'x': np.float64(-4.0), 'y': np.float64(12.0)}
"""

