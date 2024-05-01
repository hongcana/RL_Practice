
import numpy as np
import matplotlib.pyplot as plt

# n번의 실험을 거쳐서 결과를 만들어보는 python file

# 실험 횟수
runs = 200

# num of trials ( maximum vale of k )
max_k = 100

# discount rate
gamma = 0.9

# 초기 상태값 설정 ( L1 = 1, L2 = 2)
L1 = 1

# 왼쪽으로 움직일 확률
p_left = 0.5

# 확률적 정책을 설정합니다.
p_a = [p_left, 1.0-p_left] # [0.5, 0.5]

# 전체 실험 기록용 배열 초기화
all_trace_G = np.zeros((runs, max_k))

for run in range(runs):
    # 초기 상태
    l = L1

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

    all_trace_G[run] = trace_G
    print('run', run+1, ': G=', G)

# 전체 기록 평균
avg_trace_G = np.average(all_trace_G, axis= 0)

# 벨만 방정식에 구한 진짜 상태 가치
true_v = -2.25

# x축의 값을 작성
k_vals = np.arange(1, max_k+1)

plt.figure(figsize=(8, 6))
for run in range(runs):
    plt.plot(k_vals, all_trace_G[run], alpha=0.1) # 그동안 실험한 값 작성
plt.plot(k_vals, avg_trace_G, color='red', label='V(s)') # 평균 선 작성
plt.hlines(y=true_v, xmin=1, xmax=max_k+1, color='orange', linestyle='--', label='v(s)')
plt.xlabel('k')
plt.ylabel('value')
plt.suptitle('State-Value Function', fontsize=20)
plt.title(
    '$v(L'+str(L1)+')='+str(true_v) + 
    ', V(L'+str(L1)+')='+str(np.round(avg_trace_G[-1], 3)) + 
    ', \gamma='+str(gamma)+'$', 
    loc='left'
)
plt.grid()
plt.legend()
plt.show()