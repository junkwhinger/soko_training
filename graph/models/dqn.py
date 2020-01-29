import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, config):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(config.state_dim, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)

        self.out = nn.Linear(512, config.action_dim)

    def forward(self, input_tensor):
        bsz = input_tensor.size(0)

        x = self.conv1(input_tensor)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = x.view(bsz, -1)

        x = self.fc4(x)
        x = self.relu4(x)

        x = self.out(x)
        return x


class DuelingDQN(nn.Module):
    def __init__(self, config):
        super(DuelingDQN, self).__init__()
        self.config = config

        self.conv1 = nn.Conv2d(config.state_dim, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)

        self.state_out = nn.Linear(512, 1)
        self.action_out = nn.Linear(512, config.action_dim)

    def forward(self, input_tensor):

        bsz = input_tensor.size(0)

        x = self.conv1(input_tensor)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = x.view(bsz, -1)

        x = self.fc4(x)
        x = self.relu4(x)

        # state head
        v_x = self.state_out(x)

        # action head
        a_x = self.action_out(x)

        # average operator
        average_op = (1 / self.config.action_dim) * a_x

        x = v_x + (a_x - average_op)

        return x


class DuelingDQN_v2(nn.Module):
    def __init__(self, config):
        super(DuelingDQN_v2, self).__init__()
        self.config = config

        self.conv1 = nn.Conv2d(config.state_dim, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)

        self.state_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

        self.action_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, config.action_dim)
        )

    def forward(self, input_tensor):
        bsz = input_tensor.size(0)

        x = self.conv1(input_tensor)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = x.view(bsz, -1)

        # state head
        v_x = self.state_stream(x)

        # action head
        a_x = self.action_stream(x)

        # average operator
        x = v_x + (a_x - a_x.mean())

        return x

# class Config:
#     def __init__(self):
#         self.action_dim = 5
#         self.state_dim = 3
#
#
# config = Config()
# model = DQN(config)
# print(model)
# print(model(torch.randn(1, 3, 84, 84)))