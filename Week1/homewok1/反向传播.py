import numpy as np

def sigmoid_deriv(a):
    return a * (1-a)

def backward_propagation(X, y, forward_res, params,m):
    W1, W2 = params["W1"], params["W2"]
    a1, a2 = forward_res["a1"], forward_res["a2"]

    delta2 = (a2 - y) * sigmoid_deriv(a2)
    dW2 = np.dot(delta2,a1.T) / m # dL/dW2 = δ2 * a1^T / m
    db2 = np.sum(delta2,axis=1,keepdims=True) / m

    delta1 = np.dot(W2.T,delta2) * sigmoid_deriv(a1) #δ1 = W2^T·δ2 * da1/dz1
    dW1 = np.dot(delta1,X.T) / m
    db1 = np.sum(delta1,axis=1,keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

