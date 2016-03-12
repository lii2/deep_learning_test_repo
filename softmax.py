#written by Chi-Young Jeffrey Lii
input_ = [3.0, 1.0, 0.2]
    
weights = [1, 1, 1]
bias = [0, 0, 0]

# linear modeling
scores = weights*x + bias

import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    denom = sum(np.exp(x))
    
    prob = np.exp(x)/denom
    
    return prob

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
