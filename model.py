import torch.nn as nn
class Q_net(nn.Module):
    def __init__(self, in_channels = 4, num_actions = 2):
        super().__init__()
        '''
        ######################################################
        TODO: Implement Your Model
        ######################################################
        '''
        self.fc1 = nn.Linear(4,64)
        self.fc2 = nn.Linear(64,128) 
        self.fc3 = nn.Linear(128,256)
        self.fc4 = nn.Linear(256,2) 

        self.relu = nn.ReLU()  

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x