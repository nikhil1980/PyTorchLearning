""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""
A simple walk through of how to code a convolutional neural network (CNN)
using the PyTorch library. For demo, we train_old it on the very common MNIST 
dataset of handwritten digits. In this code we go through how to create 
the network as well as initialize a loss function, optimizer, check 
accuracy and more.
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
from alive_progress import alive_bar


""" 2. Create CNN """


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        """
        Here we define the layers of the network.

        Parameters:
            in_channels: the number of channels in an image, in this case 1 (grey scale images)
            num_classes: the number of classes we want to predict, in this case 10 (0-9)

        """
        super(CNN, self).__init__()
        # Add 2D Convolution layer. in_channels here is 1 for grey mode images
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # 8X28X28
        # Add Max pool layer with 2X2 kernel that skips 2 units
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 8X14X14
        # Add another 2D Convolution layer
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # 16X14X14
        # Add an FC layer Two Max pools result in 7X7
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten to 64 X 784
        x = self.fc1(x)
        return x


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
    in_channels = 1
    num_classes = 10
    learning_rate = 3e-4  # karpathy's constant
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
    model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

    """ 7. Define Loss and configure Optimizer """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"train Set length: {(len(train_loader.dataset))}")
    print(f"test Set length: {(len(test_loader.dataset))}")

    """ 8. train the network """
    for epoch in range(num_epochs):
        with alive_bar(len(train_loader), bar='bubbles', spinner='notes2', force_tty=True) as bar:
            for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

                # Get data to cuda if possible
                data = data.to(device=device)
                targets = targets.to(device=device)

                # Forward
                scores = model(data)
                bar()
                loss = criterion(scores, targets)

                # Backward
                optimizer.zero_grad()  # Set gradients to 0 for each batch
                loss.backward()

                # Gradient descent or adam step
                optimizer.step()

    """ 9. Check Accuracy on Training and test Data """
    print(f"Accuracy on training set: {check_accuracy(train_loader, model, device)*100:.2f}")
    print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")


def check_fc_2_convo2d():
    inputs = torch.tensor([[[[1., 2.],
                             [3., 4.]]]])

    print(inputs.shape)

    # Case 1. Use FC layer
    fc = torch.nn.Linear(4, 1)
    weights = torch.tensor([[1.1, 1.2, 1.3, 1.4],
                            [1.5, 1.6, 1.7, 1.8]])
    bias = torch.tensor([1.9, 2.0])
    fc.weight.data = weights
    fc.bias.data = bias
    y_fc = torch.relu(fc(inputs.view(-1, 4)))

    # 1 X 2 tensor
    print(y_fc, y_fc.shape)

    # Case 2. Use Convolutions with number of kernels equal to dataset size
    conv = torch.nn.Conv2d(in_channels=1,
                           out_channels=2,
                           kernel_size=inputs.squeeze(dim=(0)).squeeze(dim=(0)).size())  # 2X2

    print(conv.weight.size())
    print(conv.bias.size())

    conv.weight.data = weights.view(2, 1, 2, 2)
    conv.bias.data = bias

    y_conv = torch.relu(conv(inputs))
    # 1 (sample) X 2 (channels) X 1 (wd) X 1 (ht)
    print(y_conv, y_conv.shape)

    # Case 2. Convolution with 1X1 kernels with multiple channels
    conv = torch.nn.Conv2d(in_channels=4,
                           out_channels=2,
                           kernel_size=(1, 1))

    conv.weight.data = weights.view(2, 4, 1, 1)
    conv.bias.data = bias
    y_conv_1by1 = torch.relu(conv(inputs.view(1, 4, 1, 1)))

    # 1 (sample) X 2 (channels) X 1 (wd) X 1 (ht)
    print(y_conv_1by1, y_conv_1by1.shape)


if __name__ == '__main__':
    """
    Can Fully Connected Layers be Replaced by Convolutional Layers?
    
    https://sebastianraschka.com/faq/docs/fc-to-conv.html
    """
    check_fc_2_convo2d()

    # Basic Pytorch operations
    #main()



