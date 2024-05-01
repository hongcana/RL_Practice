import numpy as np
import matplotlib.pyplot as plt

max_k = 100
gamma = 0.9
L1 = 1 # 초기 상태값 설정 ( L1 = 1, L2 = 2)
s = L1
G = 0 # 수익은 0부터 시작
trace_G = []

for k in range(max_k):
    if s == 1:
        r_left, r_right = [-1.0, 1.0]

        # 최적의 행동을 찾기 : 보상을 최대화 하는
        a_idx = np.argmax([r_left, r_right]) # 최적의 행동을 찾음
        a = ['left','right'][a_idx]

        # 최적의 행동에 따른 보상을 얻음
        r = np.max([r_left, r_right])

        if a == 'left':
            pass

        elif a == 'right':
            s = 2 #L2 상태로
    
    elif s == 2:
        r_left, r_right = [0.0, -1.0]
        a_idx = np.argmax([r_left, r_right])
        a = ['left','right'][a_idx]
        r = np.max([r_left, r_right])
        if a == 'left':
            s = 1
        elif a == 'right':
            pass
    
    G += gamma**k * r

    trace_G.append(G)
print(G)

true_v = 5.26

k_vals = np.arange(1, max_k+1)

plt.figure(figsize=(8, 6))
plt.plot(k_vals, trace_G, label='$G_t = \sum_k\ \gamma^k R_{t+k}$')
plt.hlines(y=true_v, xmin=1, xmax=max_k+1, color='orange', linestyle='--', label='$v_{*}(s)$')
plt.xlabel('k')
plt.ylabel('value')
plt.suptitle('Bellman Optimality Equation', fontsize=20)
plt.title(
    '$v_{*}(L'+str(L1)+')='+str(true_v) + 
    ', V(L'+str(L1)+')='+str(np.round(trace_G[-1], 3)) + 
    ', \gamma='+str(gamma)+'$', 
    loc='left'
)
plt.grid()
plt.legend()
plt.show()
