import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-(z)))

def der_sigmoid(z):
    x = sigmoid(z)
    return  x*(1-x)


def tanh(z):
    return np.tanh(z)

def der_tanh(z):
    return 1 - np.tanh(z) ** 2


def relu(z):
    return (z>0)*(z) + ((z<0)*(z)*0.01)
    #return np.maximum(z,0)
    #return np.where(z<0, 0.01*z, z)

def der_relu(z):
    return (z>0)*np.ones(z.shape) + (z<0)*(0.01*np.ones(z.shape) )


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))
