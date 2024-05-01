import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

import sys
sys.path.append('../RL_PRACTICE')
from common.gridworld import GridWorld

def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V

env = GridWorld()
pi = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
V = defaultdict(lambda: 0)
gamma = 0.9

# 한 스탭만 수행해보고 싶으면 아래 주석 해제 후 확인 
# V = eval_onestep(pi, V, env, gamma)
# env.render_v(v=V, policy=pi)

threshold = 0.001
trace_V = [{state: 0 for state in env.states()}]
cnt = 0
while True:
    old_V = V.copy()
    V = eval_onestep(pi, V, env, gamma)
    delta = 0

    for state in V.keys():
        t = abs(V[state] - old_V[state])

        if delta < t:
            delta = t

    cnt += 1

    print('iter', cnt, ': delta', delta)
    trace_V.append(V.copy())

    if delta < threshold:
        break
env.render_v(v=V, policy=pi)


xs = env.width
ys = env.height

vmin = min(trace_V[-1].values())
vmax = max(trace_V[-1].values())

vmax = max(vmax, abs(vmin))
vmin = -1 * vmax
if vmax < 1:
    vmax = 1
if vmin > -1:
    vmin = -1

color_list = ['red', 'white', 'green']
cmap = LinearSegmentedColormap.from_list('colormap_name', color_list)

fig = plt.figure(figsize=(9, 6))
plt.suptitle('Iterative Policy Evaluation', fontsize=20)

def update(i):
    plt.cla()
    
    V = trace_V[i]
    
    v = np.zeros((env.shape))
    for state, value in V.items():
        v[state] = value
        
    plt.pcolormesh(np.flipud(v), cmap=cmap, vmin=vmin, vmax=vmax)
    
    for state in env.states():
        y, x = state
        r = env.reward_map[state]
        
        if r != 0 and r is not None:
            txt = 'R ' + str(r)
            
            if state == env.goal_state:
                txt = txt + ' (GOAL)'
            
            plt.text(x=x+0.1, y=ys-y-0.9, s=txt)
            
        if state != env.wall_state:
            plt.text(x=x+0.4, y=ys-y-0.15, s='{:12.2f}'.format(v[y, x]))
            
            actions = pi[state]
            max_actions = [k for k, v in actions.items() if v == max(actions.values())]
            
            arrows = ['↑', '↓', '←', '→']
            offsets = [(0, 0.1), (0, -0.1), (-0.1, 0), (0.1, 0)]
            
            for action in max_actions:
                arrow = arrows[action]
                offset = offsets[action]
                
                if state == env.goal_state:
                    continue
                
                plt.text(x=x+0.45+offset[0], y=ys-y-0.5+offset[1], s=arrow, fontsize=20)
        
        if state == env.wall_state:
            plt.gca().add_patch(plt.Rectangle(xy=(x, ys-y-1), width=1, height=1, fc=(0.4, 0.4, 0.4, 1.0)))
    
    
    plt.xticks(ticks=np.arange(xs))
    plt.yticks(ticks=np.arange(ys))
    plt.xlim(xmin=0, xmax=xs) 
    plt.ylim(ymin=0, ymax=ys) 
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.grid() 
    plt.title('iter:'+str(i), loc='left')

anime = FuncAnimation(fig, update, frames=len(trace_V), interval=100)
anime.save('policy_eval.gif')
