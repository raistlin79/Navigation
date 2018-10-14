import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_sizes=64, dueling_sizes=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (int_array): list of sizes of hidden layers
            dueling_sizes (int_array): list of sizes of hidden layers for dueling streams
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.num_hidden_layers = len(hidden_sizes)
        self.num_dueling_layers = len(dueling_sizes)
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 1)])
        if len(dueling_sizes):
            self.adv_fc1 = nn.Linear(hidden_sizes[-1], dueling_sizes[0])
            self.val_fc1 = nn.Linear(hidden_sizes[-1], dueling_sizes[0])
            self.adv_layers = nn.ModuleList([nn.Linear(dueling_sizes[i], dueling_sizes[i+1]) for i in range(len(dueling_sizes) - 1)])
            self.val_layers = nn.ModuleList([nn.Linear(dueling_sizes[i], dueling_sizes[i+1]) for i in range(len(dueling_sizes) - 1)])
            self.adv_out = nn.Linear(dueling_sizes[-1], action_size)
            self.val_out = nn.Linear(dueling_sizes[-1], 1)
        else:
            self.adv_out = nn.Linear(hidden_sizes[-1], action_size)
            self.val_out = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        for l in self.layers:
            x = F.relu(l(x))
        adv, val = None, None
        if self.num_dueling_layers:
            adv = F.relu(self.adv_fc1(x))
            val = F.relu(self.val_fc1(x))
            for a_l in self.adv_layers:
                adv = F.relu(a_l(adv))
            for v_l in self.val_layers:
                val = F.relu(v_l(val))
            adv = self.adv_out(adv)
            val = self.val_out(val)
        else:
            adv = self.adv_out(x)
            val = self.val_out(x)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)

        return x
