import numpy as np

def Sample(dices=2):
    x = 0
    for _ in range(dices):
        x += np.random.choice([1,2,3,4,5,6])
    return x

trial = 1000 # times of sampling

samples = []
for _ in range(trial): # 일반적인 산술평균 샘플링
    s = Sample()
    samples.append(s)

V = sum(samples) / len(samples)
print(V)

I, n = 0, 0
for _ in range(trial): # 증분(incremental)방식 샘플링 평균
    s = Sample()
    n += 1
    I += (s-I) / n
    print(I)