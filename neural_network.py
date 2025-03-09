import numpy as np
import scipy as sp
import wandb
import time

from imp_function import *


class FeedForwardNeuralNetwork:
    def __init__(self, hidden_layers, no_of_hidden_neurons, X_train, Y_train, N_train,
                  X_val, Y_val, N_val,X_test,Y_test_raw, N_test, optimizer,batch_size, 
                  weight_decay,learning_rate, max_epochs,activation, initializer,loss):

        
        self.num_classes = np.max(Y_train) + 1  # NUM_CLASSES
        self.hidden_layers = hidden_layers
        self.no_of_hidden_neurons = no_of_hidden_neurons
        self.output_layer_size = self.num_classes
        self.img_height = X_train.shape[1]
        self.img_width = X_train.shape[2]
        self.img_flattened_size = self.img_height * self.img_width
       
        self.layers = (
            [self.img_flattened_size]
            + hidden_layers * [no_of_hidden_neurons]
            + [self.output_layer_size]
        )

        self.N_train = N_train
        self.N_val = N_val
        self.N_test = N_test
        

        # to convert image pixels to row vectors
        self.X_train = np.transpose(
            X_train.reshape(
                X_train.shape[0], X_train.shape[1] * X_train.shape[2]
            )
        ) 
        self.X_test = np.transpose(
            X_test.reshape(
                X_test.shape[0], X_test.shape[1] * X_test.shape[2]
            )
        )  
        self.X_val = np.transpose(
            X_val.reshape(
                X_val.shape[0], X_val.shape[1] * X_val.shape[2]
            )
        )

        ## to convert color scale to 0-1 range from 0-255
        self.X_train = self.X_train / 255
        self.X_test = self.X_test / 255
        self.X_val = self.X_val / 255
        
        self.Y_train = self.oneHotEncode(Y_train)  # [NUM_CLASSES X NTRAIN]
        self.Y_val = self.oneHotEncode(Y_val)
        self.Y_test = self.oneHotEncode(Y_test_raw)


        self.all_Activations = {"sigmoid": sigmoid, "tanh": tanh, "ReLU": relu}
        self.all_der_of_Activation = {"sigmoid": der_sigmoid, "tanh": der_tanh, "ReLU": der_relu}

        self.all_Initializer = {"XAVIER": self.Xavier_initializer, "RANDOM": self.random_initializer}

        self.Optimizer_dict = {"SGD": self.sgdMiniBatch,"MGD": self.mgd,"NAG": self.nag,
                               "RMSPROP": self.rmsProp,"ADAM": self.adam,"NADAM": self.nadam}
        
        self.activation = self.all_Activations[activation]
        self.der_activation = self.all_der_of_Activation[activation]
        self.optimizer = self.Optimizer_dict[optimizer]
        self.initializer = self.all_Initializer[initializer]
        self.loss_function = loss
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.weights, self.biases = self.initializeNeuralNet(self.layers)


        
        
    # helper functions
    def oneHotEncode(self, Y_train):
        Ydata = np.zeros((self.num_classes, Y_train.shape[0]))
        for i in range(Y_train.shape[0]):
            value = Y_train[i]
            Ydata[int(value)][i] = 1.0
        return Ydata

    # Loss functions
    def meanSquaredErrorLoss(self, Y_true, Y_pred):
        MSE = np.mean((Y_true - Y_pred) ** 2)
        return MSE

    def crossEntropyLoss(self, Y_true, Y_pred):
        CE = [-Y_true[i] * np.log(Y_pred[i]) for i in range(len(Y_pred))]
        crossEntropy = np.mean(CE)
        return crossEntropy

    def L2RegularisationLoss(self, weight_decay):
        ALPHA = weight_decay
        return ALPHA * np.sum(
            [
                np.linalg.norm(self.weights[str(i + 1)]) ** 2
                for i in range(len(self.weights))
            ]
        )


    def accuracy(self, Y_true, Y_pred, data_size):
        Y_true_label = []
        Y_pred_label = []
        ctr = 0
        for i in range(data_size):
            Y_true_label.append(np.argmax(Y_true[:, i]))
            Y_pred_label.append(np.argmax(Y_pred[:, i]))
            if Y_true_label[i] == Y_pred_label[i]:
                ctr += 1
        accuracy = ctr / data_size
        return accuracy, Y_true_label, Y_pred_label

    def Xavier_initializer(self, size):
        in_dim = size[1]
        out_dim = size[0]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return np.random.normal(0, xavier_stddev, size=(out_dim, in_dim))

    def random_initializer(self, size):
        in_dim = size[1]
        out_dim = size[0]
        return np.random.normal(0, 1, size=(out_dim, in_dim))


    def initializeNeuralNet(self, layers):
        weights = {}
        biases = {}
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.initializer(size=[layers[l + 1], layers[l]])
            b = np.zeros((layers[l + 1], 1))
            weights[str(l + 1)] = W
            biases[str(l + 1)] = b
        return weights, biases

    def forwardPropagate(self, X_train_batch, weights, biases):
        """
        Returns the neural network given input data, weights, biases.
        Arguments:
                 : X_train_batch - input matrix
                 : Weights  - Weights matrix
                 : biases - Bias vectors 
        """
        # Number of layers = length of weight matrix + 1
        num_layers = len(weights) + 1
        # A - Preactivations
        # H - Activations
        X = X_train_batch
        H = {}
        A = {}
        H["0"] = X
        A["0"] = X
        for l in range(0, num_layers - 2):
            if l == 0:
                W = weights[str(l + 1)]
                b = biases[str(l + 1)]
                A[str(l + 1)] = np.add(np.matmul(W, X), b)
                H[str(l + 1)] = self.activation(A[str(l + 1)])
            else:
                W = weights[str(l + 1)]
                b = biases[str(l + 1)]
                A[str(l + 1)] = np.add(np.matmul(W, H[str(l)]), b)
                H[str(l + 1)] = self.activation(A[str(l + 1)])

        # Here the last layer is not activated
        W = weights[str(num_layers - 1)]
        b = biases[str(num_layers - 1)]
        A[str(num_layers - 1)] = np.add(np.matmul(W, H[str(num_layers - 2)]), b)
        # Y = softmax(A[-1])
        Y = softmax(A[str(num_layers - 1)])
        H[str(num_layers - 1)] = Y
        return Y, H, A

    def backPropagate(self, Y, H, A, Y_train_batch, weight_decay=0):
        ALPHA = weight_decay
        gradients_weights = []
        gradients_biases = []
        num_layers = len(self.layers)

        dA = {}  # Store activation gradients
        dH = {}  # Store hidden gradients
        dW = {}  # Store weight gradients
        dB = {}  # Store bias gradients

        # Compute gradient for the output layer
        if self.loss_function == "CROSS":
            dA[num_layers - 1] = -(Y_train_batch - Y)
        elif self.loss_function == "MSE":
            dA[num_layers - 1] = np.multiply(2 * (Y - Y_train_batch), np.multiply(Y, (1 - Y)))

        for l in range(num_layers - 2, -1, -1):
            dW[l + 1] = np.outer(dA[l + 1], H[str(l)]) + (ALPHA * self.weights[str(l + 1)] if ALPHA != 0 else 0)
            dB[l + 1] = dA[l + 1]

            gradients_weights.append(dW[l + 1])
            gradients_biases.append(dB[l + 1])

            if l != 0:
                dH[l] = np.matmul(self.weights[str(l + 1)].T, dA[l + 1])
                dA[l] = np.multiply(dH[l], self.der_activation(A[str(l)]))
            else:
                dH[l] = np.matmul(self.weights[str(l + 1)].T, dA[l + 1])
                dA[l] = np.multiply(dH[l], A[str(l)])

        return gradients_weights, gradients_biases


    def predict(self,X,length_dataset):
        Y_pred = []        
        for i in range(length_dataset):

            Y, H, A = self.forwardPropagate(
                X[:, i].reshape(self.img_flattened_size, 1),
                self.weights,
                self.biases,
            )

            Y_pred.append(Y.reshape(self.num_classes,))
        Y_pred = np.array(Y_pred).transpose()
        return Y_pred

