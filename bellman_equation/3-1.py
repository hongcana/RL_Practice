# numpy import
import numpy as np

# 확률 변수 셋팅
V = 6
x = np.arange(1, V+1)
print(x)

# 각 변수에 대한 확률 셋팅
p_x = np.repeat(1/V, V)
print(np.round(p_x, 3))

# 기댓값 계산
E = np.sum(x * p_x)
print(E)

############################################

