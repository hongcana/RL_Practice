# Monte Carlo with Exploring Starts (MC ES)

import numpy as np
from collections import defaultdict

# 블랙잭 게임 클래스
class Blackjack:
    def __init__(self):
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4

    def draw_card(self):
        return np.random.choice(self.deck)

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def usable_ace(self, hand):
        return 1 in hand and sum(hand) + 10 <= 21

    def total(self, hand):
        if self.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    def play_game(self, policy):
        player_hand = self.draw_hand()
        dealer_hand = self.draw_hand()

        # 플레이어의 정책에 따른 행동
        while policy(self.total(player_hand)):
            player_hand.append(self.draw_card())
            if self.total(player_hand) > 21:
                return -1  # 플레이어 패배

        # 딜러의 정책 (17 이상이 될 때까지 카드를 뽑음)
        while self.total(dealer_hand) < 17:
            dealer_hand.append(self.draw_card())
            if self.total(dealer_hand) > 21:
                return 1  # 딜러 패배

        # 승자 결정
        if self.total(player_hand) > self.total(dealer_hand):
            return 1
        elif self.total(player_hand) < self.total(dealer_hand):
            return -1
        else:
            return 0

def monte_carlo_es(episodes):
    game = Blackjack()
    
    # 가치 함수 초기화
    Q = defaultdict(lambda: np.zeros(2))
    returns = defaultdict(list)
    
    # 모든 상태-행동 쌍에 대해 임의의 정책을 시작으로 학습
    for _ in range(episodes):
        state_actions = []
        current_state = (game.total(game.draw_hand()), game.draw_card(), True)
        while current_state[0] < 21:  # 21이 넘으면 게임 종료
            action = np.random.choice([0, 1])  # 0: stop, 1: hit
            state_actions.append((current_state, action))
            if action == 0:  # 멈춤
                break
            current_state = (current_state[0] + game.draw_card(), current_state[1], current_state[2])

        reward = game.play_game(lambda x: np.random.choice([True, False]))
        
        # 가치 함수 업데이트
        for state, action in state_actions:
            returns[(state, action)].append(reward)
            Q[state][action] = np.mean(returns[(state, action)])
    
    return Q

# 10000 에피소드로 학습
Q = monte_carlo_es(10000)

# 학습된 정책 출력 예시
for state in list(Q)[:10]:
    print(state, Q[state])
