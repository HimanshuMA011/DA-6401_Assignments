import numpy as np
from keras.datasets import fashion_mnist
import wandb

# Load the Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for the Fashion-MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Initialize wandb run
wandb.init(project="fashion-mnist-visualization")


# Log multiple images
examples = []
for i in range(10):
    # Find first image of each class
    idx = np.where(train_labels == i)[0][0]
    image = wandb.Image(train_images[idx], 
                         caption= class_names[i])
    examples.append(image)

wandb.log({"class_examples": examples})