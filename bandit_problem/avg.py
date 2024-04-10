import numpy as np

np.random.seed(0) # seed 고정
rewards = []

for n in range(1,101):
    reward = np.random.rand()
    rewards.append(reward)
    Q = sum(rewards) / n
    print(Q)