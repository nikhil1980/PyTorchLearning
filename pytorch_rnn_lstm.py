""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""
Example code of a simple RNN, GRU, LSTM on the MNIST dataset.
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
from torch.utils.data import (DataLoader, )

# For Progress Bar
from alive_progress import alive_bar  # For nice progress bar!


""" 2. Create (many-to-one) RNN """


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length, device):
        """
        Constructor

        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param num_classes:
        :param sequence_length:
        :param device:
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.device = device
        # Num of batches X Number of Time Sequences X Number of Features
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)  # Number of batches time (28X256)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Recurrent neural network with GRU (many-to-one)
class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length, device):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.device = device
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Recurrent neural network with LSTM (many-to-one)
class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length, device):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
        # For Using only last hidden state
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Comment this if you use last hidden state
        # out = out.reshape(out.shape[0], -1)  # B X SEQ_LEN X HIDDEN_SIZE

        # Decode the hidden state of the last time step
        # out = self.fc(out)
        # Decode only last hidden state of the last time step
        out = self.fc(out[:, -1,: ])
        return out


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
            x = x.to(device=device).squeeze(1)
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
    input_size = 28
    hidden_size = 256
    num_layers = 2
    num_classes = 10
    # One row at a time
    sequence_length = 28
    learning_rate = 0.005
    batch_size = 64
    num_epochs = 3

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
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    #paligemma-weights = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length, device).to(device)
    #paligemma-weights = RNN_GRU(input_size, hidden_size, num_layers, num_classes, sequence_length, device).to(device)
    model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes, sequence_length, device).to(device)

    """ 7. Define Loss and configure Optimizer """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"train Set length: {(len(train_loader.dataset))}")
    print(f"test Set length: {(len(test_loader.dataset))}")

    """ 8. train the network """
    for epoch in range(num_epochs):
        with alive_bar(len(train_loader), bar='bubbles', spinner='notes2', force_tty=True) as bar:
            for batch_idx, (data, targets) in enumerate(train_loader):

                # Remove the channel dimension from image data (BX1X28X28)
                data = data.to(device=device).squeeze(1)
                targets = targets.to(device=device)

                # Forward
                scores = model(data)
                loss = criterion(scores, targets)

                # Backward
                optimizer.zero_grad()  # Set gradients to 0 for each batch
                loss.backward()

                # Gradient descent or adam step
                optimizer.step()

                # Show Progress
                bar()

    """ 9. Check Accuracy on Training and test Data """
    print(f"Accuracy on training set: {check_accuracy(train_loader, model, device)*100:.2f}")
    print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")


if __name__ == '__main__':
    main()