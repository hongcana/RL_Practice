import sys
sys.path.append('../RL_PRACTICE')
from common.gridworld import GridWorld
import numpy as np
import pandas as pd
from collections import defaultdict

class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_choices = {0: 0.25, 1: 0.25, 2: 0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_choices)
        self.V = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]

env = GridWorld()
agent = RandomAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)                # 행동 선택
        next_state, reward, done = env.step(action)     # 행동 수행
        
        agent.add(state, action, reward)      # (상태,행동,보상) 저장
        if done:
            agent.eval()
            break
        
        state = next_state
env.render_v(agent.V)