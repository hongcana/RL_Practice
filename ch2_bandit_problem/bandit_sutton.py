import numpy as np
import matplotlib.pyplot as plt


class Bandit():
    def __init__(self, arms = 10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand(): # 확률을 이기면 보상1, 아니면0
            return 1
        else:
            return 0

class Agent():
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        #self.Qs = np.zeros(action_size)
        # 긍정적 초깃값 방법
        self.Qs = np.full(action_size, 5.)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        # epsilon 확률에 따라 탐험, 탐색을 고르는 함수를 직접 구현하라.
        if np.random.rand() < self.epsilon:
            return np.random.randint(low=0, high=len(self.Qs))
        else:
            return np.argmax(self.Qs)

runs = 100
steps = 600
epsilon = 0.
all_rates = np.zeros((runs, steps))

for run in range(runs):
    bandit=Bandit()
    agent=Agent(epsilon=epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step+1))
    
    all_rates[run] = rates

avg_rates = np.average(all_rates, axis=0)

# 스텝별 승률 그래프
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.show()