import numpy as np

# Activation functions
def sigmoid(x):
    return 1/1+np.exp(-x)

def sigmoid_derivative(x):
    return x*(1-x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Constructing neural network
def build_network(input_size, hidden_layers, output_size):
    layers = [input_size] + hidden_layers + [output_size]
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.01)
        biases.append(np.zeros((1, layers[i + 1])))
    return weights, biases

# Forward propagation
def forward_propagation(X, weights, biases):
    activations = [X]
    pre_activations = []
    for i in range(len(weights) - 1):
        Z = np.dot(activations[-1], weights[i]) + biases[i]
        pre_activations.append(Z)
        A = relu(Z)
        activations.append(A)
    
    Z = np.dot(activations[-1], weights[-1]) + biases[-1]
    pre_activations.append(Z)
    A = softmax(Z)
    activations.append(A)
    
    return activations, pre_activations