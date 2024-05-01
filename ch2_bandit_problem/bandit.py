import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, arms = 10):
        self.rates = np.random.rand(arms) # set probability about each bandits(random)
    
    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0

class Agent:
    """
    Agent that select action
    according to epsilon-greedy policy.
    """
    def __init__(self, epsilon, action_size = 10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward): # estimate value of bandit
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]
        
    def get_action(self): # select action(epsilon-greedy policy)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs)) # 무작위 행동 선택
        return np.argmax(self.Qs) # 탐욕 행동 선택

runs = 200
steps = 2000
epsilon = 0.1
all_rates = np.zeros((runs, steps)) # (200, 1000) 차원 배열

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon=epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action() # 행동 선택
        reward = bandit.play(action) # 실제로 플레이하고 보상을 받음
        agent.update(action, reward) # 행동과 보상을 통해 학습
        total_reward += reward

        total_rewards.append(total_reward) # 현재까지의 보상 합 저장
        rates.append(total_reward / (step + 1)) # 현재까지의 승률 저장

    all_rates[run] = rates # 보상 결과 기록 ( 정확히는 승률 결과를 기록함 )

# Step별 평균을 구했다.
avg_rates = np.average(all_rates, axis = 0)

# 스텝별 승률 그래프
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.show()