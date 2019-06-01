import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    crossEntropy=[]
    for index in range(0,len(Y)):
        crossEntropy.append( (Y[index]*np.log(P[index])) + ( (1-Y[index])*np.log(1-P[index])))
    cross_entrop_result=(-1) * np.sum(crossEntropy)
    return cross_entrop_result
