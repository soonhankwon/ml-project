import hyperopt

print(hyperopt.__version__) # 0.2.7

from hyperopt import hp
from hyperopt import fmin, tpe, Trials
import numpy as np

# -10 ~ 10까지 1간격을 가지는 입력 변수 x와 -15 ~ 15까지 1간격으로 입력 변수 y설정
search_space = {'x': hp.quniform('x', -10, 10, 1), 'y': hp.quniform('y', -15, 15, 1)}

# 목적 함수를 생성. 변숫값과 변수 검색 공간을 가지는 딕셔너리를 인자로 받고, 특정 값을 반환
def objective_fun(search_space):
    x = search_space['x']
    y = search_space['y']
    retval = x**2 - 20*y
    return retval # return {'loss': retval, 'status': 'STATUS_OK'}

# 입력 결괏값을 저장한 Trials 객체값 생성
trial_val = Trials()

# 목적 함수의 최솟값을 반환하는 최적 입력 변숫값을 5번의 입력값 시도(max_evals=5)로 찾아냄
best_01 = fmin(fn=objective_fun, space=search_space, algo=tpe.suggest, max_evals=5, trials=trial_val, rstate=np.random.default_rng(seed=0))

best_02 = fmin(fn=objective_fun, space=search_space, algo=tpe.suggest, max_evals=20, trials=trial_val)

print('best:', best_01)
print('best:', best_02)
"""
best: {'x': np.float64(-4.0), 'y': np.float64(12.0)}
best: {'x': np.float64(5.0), 'y': np.float64(13.0)}
"""

# fmin()에 인자로 들어가는 Trials 객체의 result 속성에 파이썬 리스트로 "목적 함수 반환값"들이 저장됨
# 리스트 내부의 개별 원소는 {'loss':함수 반환값, 'status': 반환 상태값}와 같은 딕셔너리임
print(trial_val.results)
"""
[{'loss': -64.0, 'status': 'ok'}, {'loss': -184.0, 'status': 'ok'}, {'loss': 56.0, 'status': 'ok'}, {'loss': -224.0, 'status': 'ok'}, {'loss': 61.0, 'status': 'ok'}]
"""

# Trials 객체의 vals 속성에 {'입력변수명': 개별 수행 시마다 입력된 값 리스트} 형태로 저장됨
print(trial_val.vals)
"""
{'x': [np.float64(-6.0), np.float64(-4.0), np.float64(4.0), np.float64(-4.0), np.float64(9.0)], 'y': [np.float64(5.0), np.float64(10.0), np.float64(-2.0), np.float64(12.0), np.float64(1.0)]}
"""

import pandas as pd

# result에서 loss 키 값에 해당하는 밸류들을 추출하여 list 생성
losses = [loss_dict['loss'] for loss_dict in trial_val.results]

# DataFrame 으로 생성
result_df = pd.DataFrame({'x': trial_val.vals['x'], 'y': trial_val.vals['y'], 'losses': losses})
print(result_df.head(3))
"""
     x     y  losses
0 -6.0   5.0   -64.0
1 -4.0  10.0  -184.0
2  4.0  -2.0    56.0
"""