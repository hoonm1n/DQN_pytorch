# plot explore_rate
from utils import get_explore_rate
import matplotlib.pyplot as plt
import numpy as np

MIN_EXPLORE_RATE = 0.01
DECAY_CONSTANT = 500

x = list(range(10000))
y = np.array(list(get_explore_rate(i, DECAY_CONSTANT, MIN_EXPLORE_RATE) for i in x))

plt.plot(x, y)
plt.plot(x[np.where(y==y.max())[0][-1]], y[np.where(y==y.max())[0][-1]], '-bo', label=f'{np.where(y==y.max())[0][-1]}epi')
plt.plot(x[y.argmin()], y[y.argmin()], '-ro', label=f'{y.argmin()}epi')


plt.title(f'EXPLORE_RATE\nDECAY_CONSTANT:{DECAY_CONSTANT}')
plt.xlabel('Episode')
plt.ylabel('Explore_rate')
plt.legend()
plt.grid(alpha=0.3)

plt.show()