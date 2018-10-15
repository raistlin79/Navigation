import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model. DuelingQNetwork"""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (int_array): list of sizes of hidden layers
            dueling_sizes (int_array): list of sizes of hidden layers for dueling streams
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size

        # Input layer
        self.fc1 = nn.Linear(state_size, 64)
        # create hidden layers according to HIDDEN_SIZES
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)

        # create duelling layers according to DUELLING_SIZES
        self.adv_fc1 = nn.Linear(64, 64)
        self.val_fc1 = nn.Linear(64, 64)

        # Output layer
        self.adv_out = nn.Linear(64, action_size)
        self.val_out = nn.Linear(64, 1)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        adv, val = None, None
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        adv = F.relu(self.adv_fc1(x))
        val = F.relu(self.val_fc1(x))
        adv = self.adv_out(adv)
        val = self.val_out(val).expand(x.size(0), self.action_size)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)

        return x
