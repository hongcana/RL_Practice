import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

import sys
sys.path.append('../RL_PRACTICE')
from common.gridworld import GridWorld


class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions) # 정책은 4방향 0.25 확률로 기본 초기화
        self.V = defaultdict(lambda: 0) # 상태 가치 함수 V는 0으로 초기화
    
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys()) # 액션 종류 가져오기
        probs = list(action_probs.values()) # 액션 종류의 확률 가져오기

        return np.random.choice(actions, p=probs) # 확률 기반 선택
    
    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state] # 목표 지점에서의 가치 함수는 0. 왜 그럴까?

        target = reward + self.gamma * next_V
        self.V[state] += (target - self.V[state]) * self.alpha



# 하나의 에피소드 진행
env = GridWorld()
agent = TdAgent()

arrows = ['↑', '↓', '←', '→']
state = env.start_state # 환경에서 기본적인 시작 상태는 (2,0)입니다.
t = 0

while True:
    t += 1 # 한 시간단위 진행

    action = agent.get_action(state)
    next_state, reward, done = env.step(action)
    agent.eval(state,reward, next_state, done)

    print(
        't=' + str(t) + 
        ', S_t=' + str(state) + 
        ', A_t=' + arrows[action] + 
        ', S_t+1=' + str(next_state) + 
        ', R_t=' + str(reward)
    )

    if done:
        break 

    state = next_state

env.render_v(v=agent.V, policy=agent.pi)