""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""
A simple walk through of how to code a fully connected neural network
using the PyTorch library. For demo, we train it on the very common 
MNIST dataset of handwritten digits. In this code we go through
how to create the network as well as initialize a loss function,optimizer,
check accuracy and more.
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""" 1. Imports """
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules

# Gives easier dataset management by creating mini batches etc.
from torch.utils.data import (DataLoader,)

# For Progress Bar
from tqdm import tqdm  # For nice progress bar!


""" 2. Create FCN """


class NN(nn.Module):
    # Here we create our simple neural network. For more details here we are subclassing and
    # inheriting from nn.Module, this is the most general way to create your networks and
    # allows for more flexibility. I encourage you to also check out nn.Sequential which
    # would be easier to use in this scenario, but I wanted to show you something that
    # "always" works and is a general approach.
    def __init__(self, input_size, num_classes):
        """
        Here we define the layers of the network. We create two fully connected layers

        Parameters:
            input_size: the size of the dataset, in this case 784 (28x28)
            num_classes: the number of classes we want to predict, in this case 10 (0-9)

        """
        super(NN, self).__init__()
        # Our first linear layer take input_size, in this case 784 (28 X 28 images) nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in this case 10.
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, images):
        """
        images here is the mnist images, and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)

        Parameters:
            images: mnist images

        Returns:
            out: the output of the network
        """

        images = F.relu(self.fc1(images))
        images = self.fc2(images)
        return images


# Check accuracy on training & test to see how good our paligemma-weights
def check_accuracy(loader, model, device):
    """
    Check accuracy of our trained paligemma-weights given a loader and a paligemma-weights

    Parameters:
        loader: torch.utils.data.DataLoader
            A loader for the dataset you want to check accuracy on
        model: nn.Module
            The paligemma-weights you want to check accuracy on
        device: torch.device
            The device on which this code will run

    Returns:
        acc: float
            The accuracy of the paligemma-weights on the dataset given by the loader
    """

    num_correct = 0
    num_samples = 0
    model.eval()

    # We don't need to keep track of gradients here, so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:

            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            # Get to correct shape
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


def main():
    """ 3. Set Device """
    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ 4. Configure HyperParameters """
    # Hyperparameters
    input_size = 784
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 10

    """ 5. Load Data """
    train_dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="dataset/", train=False, transform=transforms.ToTensor(), download=True
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    """ 6. Initialize N/w """
    model = NN(input_size=input_size, num_classes=num_classes).to(device)

    """ 7. Define Loss and configure Optimizer """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"train Set length: {(len(train_loader.dataset))}")
    print(f"test Set length: {(len(test_loader.dataset))}")

    """ 8. train the network """
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Get to correct shape
            data = data.reshape(data.shape[0], -1)  # Flatten to 64 X 784

            # Forward
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward
            optimizer.zero_grad()  # Set gradients to 0 for each batch
            loss.backward()

            # Gradient descent or adam step
            optimizer.step()

    """ 9. Check Accuracy on Training and test Data """
    print(f"Accuracy on training set: {check_accuracy(train_loader, model, device)*100:.2f}")
    print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")


if __name__ == '__main__':
    main()

