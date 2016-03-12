#written by Chi-Young Jeffrey Lii
import numpy as np

#three vector linear model for now, make scalable later
input_ = np.array([2.0, 1.0, 0.1])
    
weights = np.array([1, 1, 1])
bias = np.array([0, 0, 0])

scores = input_*weights + bias

def softmax(x):
    #Compute softmax values for each sets of scores in x.
    denom = sum(np.exp(x))
    prob = np.exp(x)/denom
    return prob

def one_hot_encoding(x):
    #generates the correct label for each logit
    ind = np.argmax(x) 
    unit_vector = np.zeros(np.size(x))
    unit_vector[ind] = 1
    return unit_vector

def cross_entropy(logit, label):
    #generates the cross entropy between label and logit
    #never take the log of the label
    return -1*sum(np.log(logit)*label)

def average_training_loss(x):
    #calculates the average_training loss of the cross entropy
    #over the entire training set
    pass

logit = softmax(scores)
label = one_hot_encoding(logit)

print(logit, label, cross_entropy(logit,label))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
