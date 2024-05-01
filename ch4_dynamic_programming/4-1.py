import numpy as np
import matplotlib.pyplot as plt

max_iter = 100
gamma = 0.9
p_left = 0.5
V = {'L1': 0.0, 'L2': 0.0}
new_V = V.copy() # V의 복사본

trace_V = [list(V.values())] # V의 기록

for i in range(max_iter):
    new_V['L1'] = p_left * (-1 + gamma * V['L1'])\
                + (1-p_left) * (1 + gamma * V['L2'])
    new_V['L2'] = p_left * (0 + gamma * V['L1'])\
                + (1-p_left) * (-1 + gamma * V['L2'])
    
    V = new_V.copy()

    trace_V.append(list(V.values()))

    print(['V_'+str(i+1)+'('+state+')='+str(np.round(value,3)) for state, value in V.items()])

trace_V = np.array(trace_V).T # [2,100] to [100,2]


v1 = -2.25
v2 = -2.75

plt.figure(figsize=(8, 6))
plt.plot(trace_V[0], color='blue', label='$V_k(L1)$')
plt.plot(trace_V[1], color='orange', label='$V_k(L2)$')
plt.hlines(y=v1, xmin=0, xmax=max_iter, color='blue', linestyle=':', label='$v_{\pi}(L1)$')
plt.hlines(y=v2, xmin=0, xmax=max_iter, color='orange', linestyle=':', label='$v_{\pi}(L2)$')
plt.xlabel('iteration (k)')
plt.ylabel('state-value')
plt.suptitle('Iterative Policy Evaluation', fontsize=20)
plt.title('$\gamma='+str(gamma) + ', \pi='+str([p_left, 1-p_left])+'$', loc='left')
plt.grid()
plt.legend()
plt.show()