import torch
from torch import nn
import torch.nn.functional as F


class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPActor, self).__init__()

        # Initialize module list to hold all layers
        self.layers = nn.ModuleList()

        # Add the first layer (input layer)
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Add the hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Add the output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # Apply orthogonal initialization to all layers except the output layer
        for layer in self.layers[:-1]:
            nn.init.orthogonal_(layer.weight, gain=1.0)  # default gain for tanh
            nn.init.zeros_(layer.bias)

        # Orthogonal initialization with a different gain for the output layer
        nn.init.orthogonal_(self.layers[-1].weight, gain=0.01)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):

        # Apply each layer with tanh activation, except the last layer
        for layer in self.layers[:-1]:
            x = F.tanh(layer(x))

        # Apply the last layer without activation (or with tanh if required)
        x = self.layers[-1](x)

        return x


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPCritic, self).__init__()

        # Initialize module list to hold all layers
        self.layers = nn.ModuleList()

        # Add the first layer (input layer)
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Add the hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Add the output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # Apply orthogonal initialization to all layers
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight)  # default gain
            nn.init.zeros_(layer.bias)

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = F.tanh(layer(x))

            # Apply the last layer without activation (or with tanh if required)
        x = self.layers[-1](x)

        return x

class MLPEmded(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPEmded, self).__init__()

        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.hidden1.weight, gain=1.0)  # default gain for tanh
        nn.init.zeros_(self.hidden1.bias)
        nn.init.orthogonal_(self.hidden2.weight, gain=1.0)  # default gain for tanh
        nn.init.zeros_(self.hidden2.bias)
        nn.init.orthogonal_(self.output.weight, gain=1.0)  # default gain for tanh
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        embed = self.output(x)
        return embed