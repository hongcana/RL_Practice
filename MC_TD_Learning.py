# 몬테카를로 Prediction과 Temporal Difference를 구현해봅니다.

"""
Conditions -
(1) 4x4 small grid world
(2) all rewards are 1, gamma is 1
(3) all transition probabilities are 1
"""

import random
import numpy as np

class GridWorld():
    def __init__(self):
        self.x = 0
        self.y = 0
    
    def step(self, a):
        if a==0:
            self.move_right()
        elif a==1:
            self.move_left()
        elif a==2:
            self.move_up()
        elif a==3:
            self.move_down()

        reward = -1
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_right(self):
        self.y += 1
        if self.y > 3:
            self.y = 3

    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0

    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0
    
    def move_down(self):
        self.x += 1
        if self.x > 3:
            self.x = 3

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else:
            return False

    def get_state(self):
        return (self.x, self.y)

    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class Agent():
    def __init__(self):
        pass

    def select_action(self):
        coin = random.random()
        if coin < 0.25:
            action = 0
        elif coin < 0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        else:
            action = 3
        return action

def MC():
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    alpha = 0.0001

    for k in range(50000): # total 50,000 episodes
        done = False
        history = []
        while not done:
            action = agent.select_action()
            (x,y), reward, done = env.step(action)
            history.append((x,y,reward))
        env.reset()

        # Update table with data after episode.
        cum_reward = 0
        for transition in history[::-1]:
            # 방문했던 상태들을 뒤에서부터 보며 차례차례 리턴을 계산
            x, y, reward = transition
            data[x][y] = data[x][y] + alpha * (cum_reward - data[x][y])
            cum_reward = reward + gamma * cum_reward
        
    for row in data:
        row = np.round(row, 2)
        print(row)

def TD():
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    alpha = 0.01 # use large value than MC

    for k in range(50000):
        done = False
        while not done:
            x, y = env.get_state()
            action = agent.select_action()
            (x_prime, y_prime), reward, done = env.step(action)
            x_prime, y_prime = env.get_state()

            # 한 번의 step이 진행되자 마자 바로 테이블의 데이터를 업데이트 해준다.
            data[x][y] = data[x][y] + alpha * (reward+gamma*data[x_prime][y_prime]-data[x][y])
        env.reset()

    for row in data:
        row = np.round(row, 2)
        print(row)

def main():
    selc = input("TD or MC? : ")
    if selc == 'TD':
        TD()
    elif selc == "MC":
        MC()

if __name__ == "__main__":
	main()