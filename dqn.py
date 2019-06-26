from enum import Enum

import torch.nn as nn


class NetArchitecture(Enum):
    VANILLA = 'vanilla'
    DUELING = 'dueling'


class DQN(nn.Module):
    """ Simple Deep Q-Network """

    def __init__(self, state_dim, action_dim, hidden_layer_sizes=[300,300],
                 dropout_rate=0.0):
        """ Initialize a DQN Network with an arbitrary amount of linear hidden
            layers """

        super(DQN, self).__init__()
        print("Architecture: DQN")

        self.dropout_rate = dropout_rate

        # create layers
        self.layers = nn.ModuleList()
        current_input_dim = state_dim
        for layer_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(current_input_dim, layer_size))
            self.layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                self.layers.append(nn.Dropout(p=dropout_rate))
            current_input_dim = layer_size
        # output layer
        self.layers.append(nn.Linear(current_input_dim, action_dim))


    def forward(self, state_batch):
        """ Forward pass: calculate Q(state) for all actions

        Args:
            input: tensor of size batch_size x state_dim

        Returns:
            output: tensor of size batch_size x action_dim
        """

        output = state_batch
        for layer in self.layers:
            output = layer(output)
        return output


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim,
                 shared_layer_sizes=[128], value_layer_sizes=[128], 
                 advantage_layer_sizes=[128], dropout_rate=0.0):
        super(DuelingDQN, self).__init__()
        print("ARCHITECTURE: Dueling")

        self.dropout_rate = dropout_rate
        # configure layers
        self.shared_layers = nn.ModuleList()
        self.value_layers = nn.ModuleList()
        self.advantage_layers = nn.ModuleList()

        # shared layer: state_dim -> shared_layer_sizes[-1]
        shared_layer_dim = state_dim
        for layer_size in shared_layer_sizes:
            self.shared_layers.append(nn.Linear(shared_layer_dim, layer_size))
            self.shared_layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                self.shared_layers.append(nn.Dropout(p=dropout_rate))
            shared_layer_dim = layer_size
        # value layer: shared_layer_sizes[-1] -> 1
        value_layer_dim = shared_layer_dim
        for layer_size in value_layer_sizes:
            self.value_layers.append(nn.Linear(value_layer_dim, layer_size))
            self.value_layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                self.value_layers.append(nn.Dropout(p=dropout_rate))
            value_layer_dim = layer_size
        self.value_layers.append(nn.Linear(value_layer_dim, 1))
        # advantage layer: shared_layer_sizes[-1] -> actions
        advantage_layer_dim = shared_layer_dim
        for layer_size in advantage_layer_sizes:
            self.advantage_layers.append(nn.Linear(advantage_layer_dim, layer_size))
            self.advantage_layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                self.advantage_layers.append(nn.Dropout(p=dropout_rate))
            advantage_layer_dim = layer_size
        self.advantage_layers.append(nn.Linear(advantage_layer_dim, action_dim))


    def forward(self, state_batch):
        shared_output = state_batch
        # shared layer representation
        for layer in self.shared_layers:
            shared_output = layer(shared_output)
        # value stream
        value_stream = shared_output
        for layer in self.value_layers:
            value_stream = layer(value_stream)
        # advantage stream
        advantage_stream = shared_output
        for layer in self.advantage_layers:
            advantage_stream = layer(advantage_stream)
        # combine value and advantage streams into Q values
        result = value_stream + advantage_stream - advantage_stream.mean()
        return result

