import numpy as np
import scipy as sp
import wandb
import time

from imp_function import *


class FeedForwardNeuralNetwork:
    def __init__(self, hidden_layers, no_of_hidden_neurons, train_In, train_out, N_train,
                  X_val_In, val_out, N_val,test_In,test_out, N_test, optimizer,batch_size, 
                  weight_decay,learning_rate, max_epochs,activation, initializer,loss):

        
        self.num_classes = np.max(train_out) + 1  # NUM_CLASSES
        self.hidden_layers = hidden_layers
        self.no_of_hidden_neurons = no_of_hidden_neurons
        self.output_layer_size = self.num_classes
        self.img_height = train_In.shape[1]
        self.img_width = train_In.shape[2]
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
        self.train_In = np.transpose(
            train_In.reshape(
                train_In.shape[0], train_In.shape[1] * train_In.shape[2]
            )
        ) 
        self.test_In = np.transpose(
            test_In.reshape(
                test_In.shape[0], test_In.shape[1] * test_In.shape[2]
            )
        )  
        self.X_val_In = np.transpose(
            X_val_In.reshape(
                X_val_In.shape[0], X_val_In.shape[1] * X_val_In.shape[2]
            )
        )

        ## to convert color scale to 0-1 range from 0-255
        self.train_In = self.train_In / 255
        self.test_In = self.test_In / 255
        self.X_val_In = self.X_val_In / 255
        
        self.train_out = self.oneHotEncode(train_out)  # [NUM_CLASSES X NTRAIN]
        self.val_out = self.oneHotEncode(val_out)
        self.Y_test = self.oneHotEncode(test_out)


        self.all_Activations = {"sigmoid": sigmoid, "tanh": tanh, "ReLU": relu}
        self.all_der_of_Activation = {"sigmoid": der_sigmoid, "tanh": der_tanh, "ReLU": der_relu}

        self.all_Initializer = {"XAVIER": self.Xavier_initializer, "RANDOM": self.random_initializer}

        self.Optimizer_dict = {"SGD": self.sgdMB,"MGD": self.mgd,"NAG": self.nag,
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
    def oneHotEncode(self, train_out):
        Ydata = np.zeros((self.num_classes, train_out.shape[0]))
        for i in range(train_out.shape[0]):
            value = train_out[i]
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

    def L2_reg_loss(self, weight_decay):
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
        no_of_layers = len(layers)
        for l in range(0, no_of_layers - 1):
            W = self.initializer(size=[layers[l + 1], layers[l]])
            b = np.zeros((layers[l + 1], 1))
            weights[str(l + 1)] = W
            biases[str(l + 1)] = b
        return weights, biases

    def forwardPropagate(self, train_In_batch, weights, biases):
        """
        Returns the neural network given input data, weights, biases.
        Arguments:
                 : train_In_batch - input matrix
                 : Weights  - Weights matrix
                 : biases - Bias vectors 
        """
        # Number of layers = length of weight matrix + 1
        no_of_layers = len(weights) + 1
        # A - Preactivations
        # H - Activations
        X = train_In_batch
        H = {}
        A = {}
        H["0"] = X
        A["0"] = X
        for l in range(0, no_of_layers - 2):
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
        W = weights[str(no_of_layers - 1)]
        b = biases[str(no_of_layers - 1)]
        A[str(no_of_layers - 1)] = np.add(np.matmul(W, H[str(no_of_layers - 2)]), b)
        # Y = softmax(A[-1])
        Y = softmax(A[str(no_of_layers - 1)])
        H[str(no_of_layers - 1)] = Y
        return Y, H, A

    def backPropagate(self, Y, H, A, train_out_batch, weight_decay=0):
        ALPHA = weight_decay
        gradients_weights = []
        gradients_biases = []
        no_of_layers = len(self.layers)

        dA = {}  # Store activation gradients
        dH = {}  # Store hidden gradients
        dW = {}  # Store weight gradients
        dB = {}  # Store bias gradients

        # Compute gradient for the output layer
        if self.loss_function == "CROSS":
            dA[no_of_layers - 1] = -(train_out_batch - Y)
        elif self.loss_function == "MSE":
            dA[no_of_layers - 1] = np.multiply(2 * (Y - train_out_batch), np.multiply(Y, (1 - Y)))

        for l in range(no_of_layers - 2, -1, -1):
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


    def predict(self,X,data_length):
        Y_pred = []        
        for i in range(data_length):

            Y, H, A = self.forwardPropagate(
                X[:, i].reshape(self.img_flattened_size, 1),
                self.weights,
                self.biases,
            )

            Y_pred.append(Y.reshape(self.num_classes,))
        Y_pred = np.array(Y_pred).transpose()
        return Y_pred

    #Optimisers defined here onwards
    def sgd(self, epochs, data_length, learning_rate, weight_decay=0):
        
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        
        no_of_layers = len(self.layers)

        train_In = self.train_In[:, :data_length]
        train_out = self.train_out[:, :data_length]

        for epoch in range(epochs):
            start_time = time.time()
            
            idx = np.random.shuffle(np.arange(data_length))
            train_In = train_In[:, idx].reshape(self.img_flattened_size, data_length)
            train_out = train_out[:, idx].reshape(self.num_classes, data_length)
            
            LOSS = []

            
            dltaW = [
                np.zeros((self.layers[l + 1], self.layers[l]))
                for l in range(0, len(self.layers) - 1)
            ]
            deltab = [
                np.zeros((self.layers[l + 1], 1))
                for l in range(0, len(self.layers) - 1)
            ]

            for i in range(data_length):

                Y, H, A = self.forwardPropagate(
                    train_In[:, i].reshape(self.img_flattened_size, 1),
                    self.weights,
                    self.biases,
                )
                dWeights, dBiases = self.backPropagate(
                    Y, H, A, train_out[:, i].reshape(self.num_classes, 1)
                )
                dltaW = [
                    dWeights[no_of_layers - 2 - i] for i in range(no_of_layers - 1)
                ]
                deltab = [
                    dBiases[no_of_layers - 2 - i] for i in range(no_of_layers - 1)
                ]


                if self.loss_function == "MSE":
                    LOSS.append(self.meanSquaredErrorLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                        )
                elif self.loss_function == "CROSS":
                    LOSS.append(
                        self.crossEntropyLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                    )

                
                self.weights = {
                    str(i + 1): (self.weights[str(i + 1)] - learning_rate * dltaW[i])
                    for i in range(len(self.weights))
                }
                self.biases = {
                    str(i + 1): (self.biases[str(i + 1)] - learning_rate * deltab[i])
                    for i in range(len(self.biases))
                }

            elapsed = time.time() - start_time
            
            Y_pred = self.predict(self.train_In, self.N_train)
            Y_pred_val = self.predict(self.X_val_In, self.N_val)
            
            if self.loss_function == "MSE":
                val_loss_epoch = np.mean([
                    self.meanSquaredErrorLoss(
                        self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),
                    ) + self.L2_reg_loss(weight_decay)
                    for j in range(self.N_val)
                ])
            elif self.loss_function == "CROSS":
                val_loss_epoch = np.mean([
                    self.crossEntropyLoss(
                        self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),
                    ) + self.L2_reg_loss(weight_decay)
                    for j in range(self.N_val)
                ])
            
            validation_loss.append(val_loss_epoch)
            training_loss.append(np.mean(LOSS))
            training_accuracy.append(self.accuracy(train_out, Y_pred, data_length)[0])
            validation_accuracy.append(self.accuracy(self.val_out, self.predict(self.X_val_In, self.N_val), self.N_val)[0])
            
            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.3f, Validation Loss: %.3e, Validation Accuracy: %.3f, Time: %.3f, Learning Rate: %.3e"
                        % (
                            epoch,
                            training_loss[epoch],
                            training_accuracy[epoch],
                            validation_loss[epoch], 
                            validation_accuracy[epoch],
                            elapsed,
                            learning_rate,
                        )
                    )

            wandb.log({
                'loss': np.mean(LOSS), 
                'training_accuracy': training_accuracy[epoch], 
                'validation_loss': validation_loss[epoch], 
                'validation_accuracy': validation_accuracy[epoch],
                'epoch': epoch 
            })

        return training_loss, training_accuracy, validation_loss, validation_accuracy, Y_pred


      
    def sgdMB(self, epochs,data_length, batch_size, learning_rate, weight_decay = 0):

        train_In = self.train_In[:, :data_length]
        train_out = self.train_out[:, :data_length]        

        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        
        no_of_layers = len(self.layers)
        num_to_update = 0


        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(data_length))
            train_In = train_In[:, idx].reshape(self.img_flattened_size, data_length)
            train_out = train_out[:, idx].reshape(self.num_classes, data_length)
            
            LOSS = []
            #Y_pred = []
            
            dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            for i in range(data_length):
                
                Y,H,A = self.forwardPropagate(train_In[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases) 
                dWeights, dBiases = self.backPropagate(Y,H,A,train_out[:,i].reshape(self.num_classes,1))
                
                dltaW = [dWeights[no_of_layers-2 - i] + dltaW[i] for i in range(no_of_layers - 1)]
                deltab = [dBiases[no_of_layers-2 - i] + deltab[i] for i in range(no_of_layers - 1)]
                
                if self.loss_function == "MSE":
                    LOSS.append(self.meanSquaredErrorLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                        )
                elif self.loss_function == "CROSS":
                    LOSS.append(
                        self.crossEntropyLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                    )
                num_to_update +=1
                
                if int(num_to_update) % batch_size == 0:
                    
                    
                    self.weights = {str(i+1):(self.weights[str(i+1)] - learning_rate*dltaW[i]/batch_size) for i in range(len(self.weights))} 
                    self.biases = {str(i+1):(self.biases[str(i+1)] - learning_rate*deltab[i]) for i in range(len(self.biases))}
                    
                    #resetting gradient updates
                    dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            elapsed = time.time() - start_time

            Y_pred = self.predict(self.train_In, self.N_train)
            training_loss.append(np.mean(LOSS))
            training_accuracy.append(self.accuracy(train_out, Y_pred, data_length)[0])
            validation_accuracy.append(self.accuracy(self.val_out, self.predict(self.X_val_In, self.N_val), self.N_val)[0])
            Y_pred_val = self.predict(self.X_val_In, self.N_val)
            
            if self.loss_function == "MSE":
                val_loss_epoch = np.mean([self.meanSquaredErrorLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            elif self.loss_function == "CROSS":
                val_loss_epoch = np.mean([self.crossEntropyLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            
            validation_loss.append(val_loss_epoch)

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.3f, Validation Loss: %.3e, Validation Accuracy: %.3f, Time: %.3f, Learning Rate: %.3e"
                        % (
                            epoch,
                            training_loss[epoch],
                            training_accuracy[epoch],
                            validation_loss[epoch],  
                            validation_accuracy[epoch],
                            elapsed,
                            learning_rate,
                        )
                    )

            wandb.log({'loss': np.mean(LOSS), 'training_accuracy': training_accuracy[epoch],'validation_loss': validation_loss[epoch],
                'validation_accuracy': validation_accuracy[epoch],'epoch': epoch })

        return training_loss, training_accuracy, validation_loss, validation_accuracy, Y_pred



    def mgd(self, epochs,data_length, batch_size, learning_rate, weight_decay = 0):
        Beta = 0.9

        train_In = self.train_In[:, :data_length]
        train_out = self.train_out[:, :data_length]       

        
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        
        no_of_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        num_to_update = 0
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(data_length))
            train_In = train_In[:, idx].reshape(self.img_flattened_size, data_length)
            train_out = train_out[:, idx].reshape(self.num_classes, data_length)

            LOSS = []

            dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            

            for i in range(data_length):
                Y,H,A = self.forwardPropagate(self.train_In[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases) 
                dWeights, dBiases = self.backPropagate(Y,H,A,self.train_out[:,i].reshape(self.num_classes,1))
                
                dltaW = [dWeights[no_of_layers-2 - i] + dltaW[i] for i in range(no_of_layers - 1)]
                deltab = [dBiases[no_of_layers-2 - i] + deltab[i] for i in range(no_of_layers - 1)]

                if self.loss_function == "MSE":
                    LOSS.append(self.meanSquaredErrorLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                        )
                elif self.loss_function == "CROSS":
                    LOSS.append(
                        self.crossEntropyLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                    )
                num_to_update +=1
                
                if int(num_to_update) % batch_size == 0:

                    v_w = [Beta*prev_v_w[i] + learning_rate*dltaW[i]/batch_size for i in range(no_of_layers - 1)]
                    v_b = [Beta*prev_v_b[i] + learning_rate*deltab[i]/batch_size for i in range(no_of_layers - 1)]
                    
                    self.weights = {str(i+1) : (self.weights[str(i+1)] - v_w[i]) for i in range(len(self.weights))}
                    self.biases = {str(i+1): (self.biases[str(i+1)] - v_b[i]) for i in range(len(self.biases))}

                    prev_v_w = v_w
                    prev_v_b = v_b

                    #resetting gradient updates
                    dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.train_In, self.N_train)
            training_loss.append(np.mean(LOSS))
            training_accuracy.append(self.accuracy(train_out, Y_pred, data_length)[0])
            validation_accuracy.append(self.accuracy(self.val_out, self.predict(self.X_val_In, self.N_val), self.N_val)[0])
            Y_pred_val = self.predict(self.X_val_In, self.N_val)
            
            if self.loss_function == "MSE":
                val_loss_epoch = np.mean([self.meanSquaredErrorLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            elif self.loss_function == "CROSS":
                val_loss_epoch = np.mean([self.crossEntropyLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            
            validation_loss.append(val_loss_epoch)

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.3f, Validation Loss: %.3e, Validation Accuracy: %.3f, Time: %.3f, Learning Rate: %.3e"
                        % (
                            epoch,
                            training_loss[epoch],
                            training_accuracy[epoch],
                            validation_loss[epoch],  
                            validation_accuracy[epoch],
                            elapsed,
                            learning_rate,
                        )
                    )

            wandb.log({'loss': np.mean(LOSS), 'training_accuracy': training_accuracy[epoch],'validation_loss': validation_loss[epoch],
                'validation_accuracy': validation_accuracy[epoch],'epoch': epoch })

        return training_loss, training_accuracy, validation_loss, validation_accuracy, Y_pred

    

    def nag(self,epochs,data_length, batch_size,learning_rate, weight_decay = 0):
        Beta = 0.9

        train_In = self.train_In[:, :data_length]
        train_out = self.train_out[:, :data_length]        


        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        
        no_of_layers = len(self.layers)
        
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        num_to_update = 0
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(data_length))
            train_In = train_In[:, idx].reshape(self.img_flattened_size, data_length)
            train_out = train_out[:, idx].reshape(self.num_classes, data_length)

            LOSS = []
            #Y_pred = []  
            
            dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            v_w = [Beta*prev_v_w[i] for i in range(0, len(self.layers)-1)]  
            v_b = [Beta*prev_v_b[i] for i in range(0, len(self.layers)-1)]

            for i in range(data_length):
                winter = {str(i+1) : self.weights[str(i+1)] - v_w[i] for i in range(0, len(self.layers)-1)}
                binter = {str(i+1) : self.biases[str(i+1)] - v_b[i] for i in range(0, len(self.layers)-1)}
                
                Y,H,A = self.forwardPropagate(self.train_In[:,i].reshape(self.img_flattened_size,1), winter, binter) 
                dWeights, dBiases = self.backPropagate(Y,H,A,self.train_out[:,i].reshape(self.num_classes,1))
                
                dltaW = [dWeights[no_of_layers-2 - i] + dltaW[i] for i in range(no_of_layers - 1)]
                deltab = [dBiases[no_of_layers-2 - i] + deltab[i] for i in range(no_of_layers - 1)]

                #Y_pred.append(Y.reshape(self.num_classes,))
                if self.loss_function == "MSE":
                    LOSS.append(self.meanSquaredErrorLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                        )
                elif self.loss_function == "CROSS":
                    LOSS.append(
                        self.crossEntropyLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                    )

                
                num_to_update +=1
                
                if int(num_to_update) % batch_size == 0:                            

                    v_w = [Beta*prev_v_w[i] + learning_rate*dltaW[i]/batch_size for i in range(no_of_layers - 1)]
                    v_b = [Beta*prev_v_b[i] + learning_rate*deltab[i]/batch_size for i in range(no_of_layers - 1)]
        
                    self.weights ={str(i+1):self.weights[str(i+1)]  - v_w[i] for i in range(len(self.weights))}
                    self.biases = {str(i+1):self.biases[str(i+1)]  - v_b[i] for i in range(len(self.biases))}
                
                    prev_v_w = v_w
                    prev_v_b = v_b

                    dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

    
            
            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.train_In, self.N_train)
            training_loss.append(np.mean(LOSS))
            training_accuracy.append(self.accuracy(train_out, Y_pred, data_length)[0])
            validation_accuracy.append(self.accuracy(self.val_out, self.predict(self.X_val_In, self.N_val), self.N_val)[0])
            Y_pred_val = self.predict(self.X_val_In, self.N_val)
            
            if self.loss_function == "MSE":
                val_loss_epoch = np.mean([self.meanSquaredErrorLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            elif self.loss_function == "CROSS":
                val_loss_epoch = np.mean([self.crossEntropyLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            
            validation_loss.append(val_loss_epoch)

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.3f, Validation Loss: %.3e, Validation Accuracy: %.3f, Time: %.3f, Learning Rate: %.3e"
                        % (
                            epoch,
                            training_loss[epoch],
                            training_accuracy[epoch],
                            validation_loss[epoch],  
                            validation_accuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )

            wandb.log({'loss': np.mean(LOSS), 'training_accuracy': training_accuracy[epoch],'validation_loss': validation_loss[epoch],
                'validation_accuracy': validation_accuracy[epoch],'epoch': epoch })

        return training_loss, training_accuracy, validation_loss, validation_accuracy, Y_pred
    

    
    def rmsProp(self, epochs,data_length, batch_size, learning_rate, weight_decay = 0):


        train_In = self.train_In[:, :data_length]
        train_out = self.train_out[:, :data_length]        

        
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        
        no_of_layers = len(self.layers)
        EPS, Beta = 1e-8, 0.9
        
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        num_to_update = 0        
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(data_length))
            train_In = train_In[:, idx].reshape(self.img_flattened_size, data_length)
            train_out = train_out[:, idx].reshape(self.num_classes, data_length)


            LOSS = []
            #Y_pred = []
                        
            dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            for i in range(data_length):
            
                Y,H,A = self.forwardPropagate(self.train_In[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases) 
                dWeights, dBiases = self.backPropagate(Y,H,A,self.train_out[:,i].reshape(self.num_classes,1))
            
                dltaW = [dWeights[no_of_layers-2 - i] + dltaW[i] for i in range(no_of_layers - 1)]
                deltab = [dBiases[no_of_layers-2 - i] + deltab[i] for i in range(no_of_layers - 1)]
                
                #Y_pred.append(Y.reshape(self.num_classes,))
                if self.loss_function == "MSE":
                    LOSS.append(self.meanSquaredErrorLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                        )
                elif self.loss_function == "CROSS":
                    LOSS.append(
                        self.crossEntropyLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                    )

                num_to_update +=1
                
                if int(num_to_update) % batch_size == 0:
                
                    v_w = [Beta*v_w[i] + (1-Beta)*(dltaW[i])**2 for i in range(no_of_layers - 1)]
                    v_b = [Beta*v_b[i] + (1-Beta)*(deltab[i])**2 for i in range(no_of_layers - 1)]

                    self.weights = {str(i+1):self.weights[str(i+1)]  - dltaW[i]*(learning_rate/np.sqrt(v_w[i]+EPS)) for i in range(len(self.weights))} 
                    self.biases = {str(i+1):self.biases[str(i+1)]  - deltab[i]*(learning_rate/np.sqrt(v_b[i]+EPS)) for i in range(len(self.biases))}

                    dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
    
            
            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.train_In, self.N_train)
            training_loss.append(np.mean(LOSS))
            training_accuracy.append(self.accuracy(train_out, Y_pred, data_length)[0])
            validation_accuracy.append(self.accuracy(self.val_out, self.predict(self.X_val_In, self.N_val), self.N_val)[0])
            Y_pred_val = self.predict(self.X_val_In, self.N_val)
            
            if self.loss_function == "MSE":
                val_loss_epoch = np.mean([self.meanSquaredErrorLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            elif self.loss_function == "CROSS":
                val_loss_epoch = np.mean([self.crossEntropyLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            
            validation_loss.append(val_loss_epoch)

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.3f, Validation Loss: %.3e, Validation Accuracy: %.3f, Time: %.3f, Learning Rate: %.3e"
                        % (
                            epoch,
                            training_loss[epoch],
                            training_accuracy[epoch],
                            validation_loss[epoch],  
                            validation_accuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )

            wandb.log({'loss': np.mean(LOSS), 'training_accuracy': training_accuracy[epoch],'validation_loss': validation_loss[epoch],
                'validation_accuracy': validation_accuracy[epoch],'epoch': epoch })

        return training_loss, training_accuracy, validation_loss, validation_accuracy, Y_pred



    def adam(self, epochs,data_length, batch_size, learning_rate, weight_decay = 0):
        
        train_In = self.train_In[:, :data_length]
        train_out = self.train_out[:, :data_length]        

        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        no_of_layers = len(self.layers)
        EPS, Beta1, Beta2 = 1e-8, 0.9, 0.99
        
        m_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]        
        
        m_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        v_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]   
        
        num_to_update = 0 
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(data_length))
            train_In = train_In[:, idx].reshape(self.img_flattened_size, data_length)
            train_out = train_out[:, idx].reshape(self.num_classes, data_length)


            LOSS = []
            #Y_pred = []
            
            dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
           
            for i in range(data_length):
                Y,H,A = self.forwardPropagate(self.train_In[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases) 
                dWeights, dBiases = self.backPropagate(Y,H,A,self.train_out[:,i].reshape(self.num_classes,1))
                
                dltaW = [dWeights[no_of_layers-2 - i] + dltaW[i] for i in range(no_of_layers - 1)]
                deltab = [dBiases[no_of_layers-2 - i] + deltab[i] for i in range(no_of_layers - 1)]

                #Y_pred.append(Y.reshape(self.num_classes,))
                if self.loss_function == "MSE":
                    LOSS.append(self.meanSquaredErrorLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                        )
                elif self.loss_function == "CROSS":
                    LOSS.append(
                        self.crossEntropyLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                    )

                
                num_to_update += 1
                ctr = 0
                if int(num_to_update) % batch_size == 0:
                    ctr += 1
                
                    m_w = [Beta1*m_w[i] + (1-Beta1)*dltaW[i] for i in range(no_of_layers - 1)]
                    m_b = [Beta1*m_b[i] + (1-Beta1)*deltab[i] for i in range(no_of_layers - 1)]
                
                    v_w = [Beta2*v_w[i] + (1-Beta2)*(dltaW[i])**2 for i in range(no_of_layers - 1)]
                    v_b = [Beta2*v_b[i] + (1-Beta2)*(deltab[i])**2 for i in range(no_of_layers - 1)]
                    
                    m_w_hat = [m_w[i]/(1-Beta1**(epoch+1)) for i in range(no_of_layers - 1)]
                    m_b_hat = [m_b[i]/(1-Beta1**(epoch+1)) for i in range(no_of_layers - 1)]            
                
                    v_w_hat = [v_w[i]/(1-Beta2**(epoch+1)) for i in range(no_of_layers - 1)]
                    v_b_hat = [v_b[i]/(1-Beta2**(epoch+1)) for i in range(no_of_layers - 1)]
                
                    self.weights = {str(i+1):self.weights[str(i+1)] - (learning_rate/np.sqrt(v_w[i]+EPS))*m_w_hat[i] for i in range(len(self.weights))} 
                    self.biases = {str(i+1):self.biases[str(i+1)] - (learning_rate/np.sqrt(v_b[i]+EPS))*m_b_hat[i] for i in range(len(self.biases))}

                    dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]


            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.train_In, self.N_train)
            training_loss.append(np.mean(LOSS))
            training_accuracy.append(self.accuracy(train_out, Y_pred, data_length)[0])
            validation_accuracy.append(self.accuracy(self.val_out, self.predict(self.X_val_In, self.N_val), self.N_val)[0])
            
            Y_pred_val = self.predict(self.X_val_In, self.N_val)
            if self.loss_function == "MSE":
                val_loss_epoch = np.mean([self.meanSquaredErrorLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            elif self.loss_function == "CROSS":
                val_loss_epoch = np.mean([self.crossEntropyLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            
            validation_loss.append(val_loss_epoch)

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.3f, Validation Loss: %.3e, Validation Accuracy: %.3f, Time: %.3f, Learning Rate: %.3e"
                        % (
                            epoch,
                            training_loss[epoch],
                            training_accuracy[epoch],
                            validation_loss[epoch],  
                            validation_accuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )

            wandb.log({'loss': np.mean(LOSS), 'training_accuracy': training_accuracy[epoch],'validation_loss': validation_loss[epoch],
                'validation_accuracy': validation_accuracy[epoch],'epoch': epoch })

        return training_loss, training_accuracy, validation_loss, validation_accuracy, Y_pred


    
    def nadam(self, epochs,data_length, batch_size, learning_rate, weight_decay = 0):

        train_In = self.train_In[:, :data_length]
        train_out = self.train_out[:, :data_length]        

        
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        no_of_layers = len(self.layers)
        
        Beta, EPS, Beta1, Beta2 = 0.9, 1e-8, 0.9, 0.99

        m_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]        

        m_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        v_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)] 

        num_to_update = 0 
        
        
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(data_length))
            train_In = train_In[:, idx].reshape(self.img_flattened_size, data_length)
            train_out = train_out[:, idx].reshape(self.num_classes, data_length)

            LOSS = []
            #Y_pred = []

            dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            for i in range(data_length):

                Y,H,A = self.forwardPropagate(self.train_In[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases) 
                dWeights, dBiases = self.backPropagate(Y,H,A,self.train_out[:,i].reshape(self.num_classes,1))

                dltaW = [dWeights[no_of_layers-2 - i] + dltaW[i] for i in range(no_of_layers - 1)]
                deltab = [dBiases[no_of_layers-2 - i] + deltab[i] for i in range(no_of_layers - 1)]

                #Y_pred.append(Y.reshape(self.num_classes,))
                if self.loss_function == "MSE":
                    LOSS.append(self.meanSquaredErrorLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                        )
                elif self.loss_function == "CROSS":
                    LOSS.append(
                        self.crossEntropyLoss(
                            self.train_out[:, i].reshape(self.num_classes, 1), Y
                        )
                        + self.L2_reg_loss(weight_decay)
                    )

                num_to_update += 1
                
                if num_to_update % batch_size == 0:
                    
                    m_w = [Beta1*m_w[i] + (1-Beta1)*dltaW[i] for i in range(no_of_layers - 1)]
                    m_b = [Beta1*m_b[i] + (1-Beta1)*deltab[i] for i in range(no_of_layers - 1)]
                    
                    v_w = [Beta2*v_w[i] + (1-Beta2)*(dltaW[i])**2 for i in range(no_of_layers - 1)]
                    v_b = [Beta2*v_b[i] + (1-Beta2)*(deltab[i])**2 for i in range(no_of_layers - 1)]
                    
                    m_w_hat = [m_w[i]/(1-Beta1**(epoch+1)) for i in range(no_of_layers - 1)]
                    m_b_hat = [m_b[i]/(1-Beta1**(epoch+1)) for i in range(no_of_layers - 1)]            
                    
                    v_w_hat = [v_w[i]/(1-Beta2**(epoch+1)) for i in range(no_of_layers - 1)]
                    v_b_hat = [v_b[i]/(1-Beta2**(epoch+1)) for i in range(no_of_layers - 1)]
                    
                    self.weights = {str(i+1):self.weights[str(i+1)] - (learning_rate/(np.sqrt(v_w_hat[i])+EPS))*(Beta1*m_w_hat[i]+ (1-Beta1)*dltaW[i]) for i in range(len(self.weights))} 
                    self.biases = {str(i+1):self.biases[str(i+1)] - (learning_rate/(np.sqrt(v_b_hat[i])+EPS))*(Beta1*m_b_hat[i] + (1-Beta1)*deltab[i]) for i in range(len(self.biases))}

                    dltaW = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
             
            elapsed = time.time() - start_time

            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.train_In, self.N_train)
            training_loss.append(np.mean(LOSS))
            training_accuracy.append(self.accuracy(train_out, Y_pred, data_length)[0])
            validation_accuracy.append(self.accuracy(self.val_out, self.predict(self.X_val_In, self.N_val), self.N_val)[0])

            Y_pred_val = self.predict(self.X_val_In, self.N_val)
            if self.loss_function == "MSE":
                val_loss_epoch = np.mean([self.meanSquaredErrorLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            elif self.loss_function == "CROSS":
                val_loss_epoch = np.mean([self.crossEntropyLoss(self.val_out[:, j].reshape(self.num_classes, 1),
                        Y_pred_val[:, j].reshape(self.num_classes, 1),) + self.L2_reg_loss(weight_decay)for j in range(self.N_val)])
            
            validation_loss.append(val_loss_epoch)

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.3f, Validation Loss: %.3e, Validation Accuracy: %.3f, Time: %.3f, Learning Rate: %.3e"
                        % (
                            epoch,
                            training_loss[epoch],
                            training_accuracy[epoch],
                            validation_loss[epoch],  
                            validation_accuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )

            wandb.log({'loss': np.mean(LOSS), 'training_accuracy': training_accuracy[epoch],'validation_loss': validation_loss[epoch],
                'validation_accuracy': validation_accuracy[epoch],'epoch': epoch })

        return training_loss, training_accuracy, validation_loss, validation_accuracy, Y_pred
        