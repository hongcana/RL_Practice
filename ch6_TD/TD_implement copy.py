import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

import sys
sys.path.append('../RL_PRACTICE')
from common.gridworld import GridWorld


class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions) # 정책은 4방향 0.25 확률로 기본 초기화
        self.V = defaultdict(lambda: 0) # 상태 가치 함수 V는 0으로 초기화
    
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys()) # 액션 종류 가져오기
        probs = list(action_probs.values()) # 액션 종류의 확률 가져오기

        return np.random.choice(actions, p=probs) # 확률 기반 선택
    
    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state] # 목표 지점에서의 가치 함수는 0. 왜 그럴까?

        target = reward + self.gamma * next_V
        self.V[state] += (target - self.V[state]) * self.alpha



# 環境・エージェントのインスタンスを作成
env = GridWorld()
agent = TdAgent()

# エピソード数を指定
episodes = 1000

# 推移の可視化用のリストを初期化
trace_V = [{state: agent.V[state] for state in env.states()}] # 初期値を記録

# 繰り返しシミュレーション
for episode in range(episodes):
    # 状態を初期化
    state = env.reset()
    
    # 時刻(試行回数)を初期化
    t = 0
    
    # 1エピソードのシミュレーション
    while True:
        # 時刻をカウント
        t += 1
        
        # ランダムに行動を決定
        action = agent.get_action(state)
        
        # サンプルデータを取得
        next_state, reward, done = env.step(action)
        
        # 現在の状態の状態価値関数を更新:式(6.9)
        agent.eval(state, reward, next_state, done)
        
        # ゴールに着いた場合
        if done:
            # 更新値を記録
            trace_V.append(agent.V.copy())
            
            # 総時刻を表示
            print('episode '+str(episode+1) + ': T='+str(t))
            
            # エピソードを終了
            break
        
        # 状態を更新
        state = next_state

# グリッドマップのサイズを取得
xs = env.width
ys = env.height

# 状態価値の最大値・最小値を取得
vmax = max([max(trace_V[i].values()) for i in range(len(trace_V))])
vmin = min([min(trace_V[i].values()) for i in range(len(trace_V))])

# 色付け用に最大値・最小値を再設定
vmax = max(vmax, abs(vmin))
vmin = -1 * vmax
vmax = 1 if vmax < 1 else vmax
vmin = -1 if vmin > -1 else vmin

# カラーマップを設定
color_list = ['red', 'white', 'green']
cmap = LinearSegmentedColormap.from_list('colormap_name', color_list)

# 図を初期化
fig = plt.figure(figsize=(10, 7.5), facecolor='white') # 図の設定
plt.suptitle('TD Method', fontsize=20) # 全体のタイトル

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目の更新値を取得
    pi = agent.pi
    V = trace_V[i]
    
    # ディクショナリを配列に変換
    v = np.zeros((env.shape))
    for state, value in V.items():
        v[state] = value
        
    # 状態価値のヒートマップを描画
    plt.pcolormesh(np.flipud(v), cmap=cmap, vmin=vmin, vmax=vmax) # ヒートマップ
    
    # マス(状態)ごとに処理
    for state in env.states():
        # インデックスを取得
        y, x = state
        
        # 報酬を抽出
        r = env.reward_map[state]
        
        # 報酬がある場合
        if r != 0 and r is not None:
            # 報酬ラベル用の文字列を作成
            txt = 'R ' + str(r)
            
            # ゴールの場合
            if state == env.goal_state:
                # 報酬ラベルにゴールを追加
                txt = txt + ' (GOAL)'
            
            # 報酬ラベルを描画
            plt.text(x=x+0.1, y=ys-y-0.9, s=txt, 
                     ha='left', va='bottom', fontsize=15)
            
        # 壁以外の場合
        if state != env.wall_state:
            # 状態価値ラベルを描画
            plt.text(x=x+0.9, y=ys-y-0.1, s=str(np.round(v[y, x], 3)), 
                     ha='right', va='top', fontsize=15)
            
            # 確率論的方策を抽出
            actions = pi[state]
            
            # 確率が最大の行動を抽出
            max_actions = [k for k, v in actions.items() if v == max(actions.values())]
            
            # 矢印の描画用のリストを作成
            arrows = ['↑', '↓', '←', '→']
            offsets = [(0, 0.1), (0, -0.1), (-0.1, 0), (0.1, 0)]
            
            # 行動ごとに処理
            for action in max_actions:
                # 矢印の描画用の値を抽出
                arrow = arrows[action]
                offset = offsets[action]
                
                # ゴールの場合
                if state == env.goal_state:
                    # 描画せず次の状態へ
                    continue
                
                # 方策ラベル(矢印)を描画
                plt.text(x=x+0.5+offset[0], y=ys-y-0.5+offset[1], s=arrow, 
                         ha='center', va='center', size=20)
        
        # 壁の場合
        if state == env.wall_state:
            # 壁を描画
            rect = plt.Rectangle(xy=(x, ys-y-1), width=1, height=1, 
                                 fc=(0.4, 0.4, 0.4, 1.0)) # 長方形を作成
            plt.gca().add_patch(rect) # 重ねて描画
    
    # グラフの設定
    plt.xticks(ticks=np.arange(xs)) # x軸の目盛位置
    plt.yticks(ticks=np.arange(ys), labels=ys-np.arange(ys)-1) # y軸の目盛位置
    plt.xlim(xmin=0, xmax=xs) # x軸の範囲
    plt.ylim(ymin=0, ymax=ys) # y軸の範囲
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False) # 軸ラベル
    plt.grid() # グリッド線
    plt.title('episode:'+str(i), loc='left') # タイトル

# gif画像を作成
anime = FuncAnimation(fig=fig, func=update, frames=len(trace_V), interval=50)

# gif画像を保存
anime.save('ch6_1.gif')

# 状態価値関数の推移を作図
plt.figure(figsize=(12, 9), facecolor='white')

# 状態ごとに推移を作図
for state in env.states():
    # マスのインデックスを取得    
    h, w = state
    
    # 更新値を抽出
    v_vals = [trace_V[i][state] for i in range(episodes+1)]
    
    # 推移を描画
    plt.plot(np.arange(episodes+1), v_vals, 
             label='$V_i(L_{'+str(h)+','+str(w)+'})$') # 各状態の価値の推移
plt.xlabel('episode')
plt.ylabel('state-value')
plt.suptitle('TD Method', fontsize=20)
plt.title('$\gamma='+str(agent.gamma) + ', \\alpha='+str(agent.alpha)+'$', loc='left')
plt.grid()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
