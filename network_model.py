import torch
from torch import nn

class LinearNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        """Network structure is defined here
        """
        super(LinearNetwork, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        self.fc_in = nn.Linear(self.input_size, 64)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, 256)
        self.fc_out = nn.Linear(256, self.num_actions)

    def forward(self, s_input):
        x = self.fc_in(s_input)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x


class ConvNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        """Network structure is defined here
        """
        super(ConvNetwork, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        self.conv1 = nn.Conv3d(1, 128, 3, stride=1)
        self.conv2 = nn.Conv3d(128, 256, 3, stride=1)
        self.conv1_1 = nn.Conv3d(1, 256, 5, stride=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(3, 32)
        self.fc3 = nn.Linear(3, 32)
        self.fc4 = nn.Linear(576, 128)
        self.fc_out = nn.Linear(128, self.num_actions)

    def forward(self, s_input):
        (state, loc, dest) = s_input
        state = state.unsqueeze(1)
        x_state_fe = self.conv1(state)
        x_state_fe = self.relu(x_state_fe)
        x_state_fe = self.conv2(x_state_fe)
        x_state_fe = self.relu(x_state_fe)
        x_state_fe_1 = self.conv1_1(state)
        x_state_fe_1 = self.relu(x_state_fe_1)
        x_state_fe = x_state_fe + x_state_fe_1
        x_state = self.fc1(x_state_fe.view(state.shape[0], -1))
        x_state = self.relu(x_state)
        x_loc = self.fc2(loc)
        x_loc = self.relu(x_loc)
        x_dest = self.fc3(dest)
        x_dest = self.relu(x_dest)
        x = torch.cat([x_state, x_loc, x_dest], -1)
        x = self.fc4(x)
        x = self.relu(x)
        out = self.fc_out(x)
        return out
