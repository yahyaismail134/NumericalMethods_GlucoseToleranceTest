import torch
import torch.nn as nn

class ApoptosisPINN(nn.Module):
    def __init__(self):
        super(ApoptosisPINN, self).__init__()
        
        # 1 Input (time), 6 Outputs (protein concentrations)
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 6),
            nn.Softplus()
        )

    def forward(self, t):
        return self.net(t)