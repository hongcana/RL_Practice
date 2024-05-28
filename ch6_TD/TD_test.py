import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

import sys
sys.path.append('../RL_PRACTICE')
from common.gridworld import GridWorld


env = GridWorld()

V = {state: np.random.rand() for state in env.states()}
print(list(V.keys()))
print(np.round(list(V.values()), 3))

next_state = (0, 2)
done = False # 다음 상태가 종결이 아님을 설정

if done:
    next_V = 0 # 종결 상태에서는 가치함수는 0
else:
    next_V = V[next_state] # next_V는 다음 상태의 가치
print(next_V) # 동작 확인

# 할인율 
gamma = 0.9

# 학습률(알파)
alpha = 0.01

# 시작상태
state = (0, 1)
print(V[state]) # 갱신 전 출력

# 보상
reward = 1

# TD target 정의
target = reward + gamma * next_V

# 가치 함수 추정치 갱신
V[state] += (target - V[state]) * alpha
print(V[state]) # 갱신 후 출력