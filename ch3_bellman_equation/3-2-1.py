
import numpy as np
import matplotlib.pyplot as plt

# num of trials ( maximum vale of k )
max_k = 100

# discount rate
gamma = 0.9

# 초기 상태값 설정 ( L1 = 1, L2 = 2)
L1 = 1

# 초기 상태
l = L1

# 왼쪽으로 움직일 확률
p_left = 0.5

# 확률적 정책을 설정합니다.
p_a = [p_left, 1.0-p_left] # [0.5, 0.5]

# 초기 return을 설정합니다.
G = 0

# 기록 list
trace_G = []

for k in range(max_k):

    # np.random.choice로 값을 랜덤하게 생성
    a = np.random.choice(['left','right'], size =1, p=p_a).item()

    # if the current state is L1
    if l == 1:
        if a == 'left':
            r = -1.0  # 초기 L1에서 left로 가면 벽이므로 보상은 -1

        elif a == 'right':
            l = 2 # 상태값 업데이트(L2)
            r = 1.0 # 보상 줌

    elif l == 2:
        if a == 'left':
            r = 0.0
            l = 1
        if a == 'right':
            r = -1.0 # 벽

    # return을 계산합니다.
    G += gamma**k * r

    # Gamma 값을 기록지에 기록합니다.
    trace_G.append(G)

# 최종 결과 확인 (매번 달라짐)
print(G)

# Specify true state value # 시작이 L1일 때
v_s = -2.25

# create x-axis values
k_vals = np.arange( 1, max_k+1)

# Plot the experimental results
plt.figure(figsize=(8,6))
plt.plot(k_vals, trace_G, label= '$G_t$')
# 벨만 방정식에서 구한 v_pi(L1) 상태가치 정답 : true state value
plt.hlines(y=v_s, xmin = 1, xmax=max_k+1, color='orange', linestyles='--', label='v(s)')
plt.xlabel('k')
plt.ylabel('value')
plt.suptitle('$G_t = \sum_k\ \gamma^k R_{t+k}$', fontsize = 20)
plt.title('$S_t=L' + str (L1) + ', p(a | s)=' + str (p_a) + ', \gamma=' + str (gamma)+ '$' , loc='left')
plt.grid()
plt.legend()
plt.show()