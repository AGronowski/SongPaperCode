import numpy as np

#y_ and u are independent
def demographic_parity(y_logits, u):
    #this is equivalent to (p > 0.5) where p is the probability of the positive event of the Bernoulli distribution
    #y_ is now the predicted labels
    y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    g, uc = np.zeros([2]), np.zeros([2])
    for i in range(u.shape[0]):
        #binary sensitive attribute is positive
        if u[i] > 0:
            g[1] += y_[i]
            uc[1] += 1
        else:
            g[0] += y_[i]
            uc[0] += 1
    #uc[0] is number of examples where sensitive attribute is 0
    #uc[1] "" is 1

    #this division is dividing the sum by the number of examples to find the mean
    g = g / uc
    return np.abs(g[0] - g[1])

#y_ and u are independent given y
def equalized_odds(y, y_logits, u):
    y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    g = np.zeros([2, 2])
    uc = np.zeros([2, 2])
    for i in range(u.shape[0]):
        if u[i] > 0:
            g[int(y[i])][1] += y_[i]
            uc[int(y[i])][1] += 1
        else:
            g[int(y[i])][0] += y_[i]
            uc[int(y[i])][0] += 1
    g = g / uc
    #first term: given y =0, second term: given y=1
    return np.abs(g[0, 1] - g[0, 0]) + np.abs(g[1, 1] - g[1, 0])

#y_ and u are independent given y=0
def equalizied_opportunity(y, y_logits, u):
    y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    g, uc = np.zeros([2]), np.zeros([2])
    for i in range(u.shape[0]):
        #y = 0
        if y[i] < 0.999:
            #leave current iteration of for loop
            continue
        if u[i] > 0:
            g[1] += y_[i]
            uc[1] += 1
        else:
            g[0] += y_[i]
            uc[0] += 1
    g = g / uc
    #here only look at where y=1, where y=0 is ignored
    return np.abs(g[0] - g[1])


def accuracy(y, y_logits):
    y_ = (y_logits > 0.0).astype(np.float32)
    #number of correct predictions divided by total predictions
    return np.mean((y_ == y).astype(np.float32))
