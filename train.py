import argparse
import numpy as np
import wandb
from neural_network import NeuralNetwork


def parse_args():
    parser = argparse.ArgumentParser(description='Building a feedforward Multi-layer neural network on Fashion-MNIST Dataset')
    parser.add_argument('-wp', '--wandb_project', type=str, default='fashion-mnist-visualization')
    parser.add_argument('-we', '--wandb_entity', type=str, default='ma23c011-indian-institute-of-technology-madras')
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-l', '--loss', type=str, default='CROSS', choices=['MSE', 'CROSS'])
    parser.add_argument('-o', '--optimizer', type=str, default='ADAM', choices=["SGD", "MGD", "NAG", "RMSPROP", "ADAM","NADAM"])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-m', '--momentum', type=float, default=0.9)
    parser.add_argument('-beta', '--beta', type=float, default=0.9)
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9)
    parser.add_argument('-beta2', '--beta2', type=float, default=0.99)
    parser.add_argument('-eps', '--epsilon', type=float, default=0.00000001)
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-w_i', '--weight_init', type=str, default='XAVIER', choices=["RANDOM", "XAVIER"])
    parser.add_argument('-nhl', '--num_layers', type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', type=int, default=64)
    parser.add_argument('-a', '--activation', type=str, default='sigmoid', choices=['sigmoid', 'tanh', 'ReLU'])
    
    return parser.parse_args()

def train():    
    args = parse_args()
    if args.dataset == "mnist":
        from keras.datasets import fashion_mnist
        (trainIn, trainOut), (testIn, testOut) = mnist.load_data()
    elif args.dataset == "fashion_mnist":
        from keras.datasets import mnist
        (trainIn, trainOut), (testIn, testOut) = fashion_mnist.load_data()  

    N_train_full = trainOut.shape[0]
    N_train = int(0.9*N_train_full)
    N_validation = int(0.1 * trainOut.shape[0])
    N_test = testOut.shape[0]


    idx  = np.random.choice(trainOut.shape[0], N_train_full, replace=False)
    idx2 = np.random.choice(testOut.shape[0], N_test, replace=False)

    trainInFull = trainIn[idx, :]
    trainOutFull = trainOut[idx]

    trainIn = trainInFull[:N_train,:]
    trainOut = trainOutFull[:N_train]

    validIn = trainInFull[N_train:, :]
    validOut = trainOutFull[N_train:]    

    testIn = testIn[idx2, :]
    testOut = testOut[idx2]

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    wandb.run.name = "hl_" + str(args.num_layers) + "_hn_" + str(args.hidden_size) + "_opt_" + args.optimize + "_act_" + args.activation + "_lr_" + str(args.learning_rate) + "_bs_"+str(args.batch_size) + "_init_" + args.weight_init + "_ep_"+ str(args.epochs)+ "_l2_" + str(args.weight_init) 

    FFNN = NeuralNetwork(
        hidden_layers=args.num_layers,
        no_of_hidden_neurons=args.hidden_size,
        train_In=trainIn,
        train_out=trainOut,
        N_train=N_train,
        X_val_In=validIn,
        val_out=validOut,
        N_val=N_validation,
        test_In=testIn,
        test_out=testOut,
        N_test=N_test,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        activation=args.activation,
        initializer=args.weight_init,
        loss=args.loss,
        Beta=args.beta,
        Beta1=args.beta1,
        Beta2=args.beta2,
        epsilon=args.epsilon
    )
    
    training_loss, training_accuracy, validation_loss, validation_accuracy, Y_pred = FFNN.optimizer(FFNN.max_epochs, FFNN.N_train, 
                                                                                                    FFNN.batch_size, FFNN.learning_rate)
    Y_test_pred = FFNN.predict(FFNN.test_In,FFNN.N_test)
    testing_accuracy = FFNN.accuracy(FFNN.Y_test,Y_test_pred,FFNN.N_test)
    print(f"Accuracy on testing data is {testing_accuracy[0]}")
    return Y_test_pred

if __name__ == "__main__":
    train()
