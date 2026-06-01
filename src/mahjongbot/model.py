import torch
from torch import nn

class rescnn(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self._resconv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self._resconv(x)

class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            # Input tensors are shaped as [batch, 71, 4, 9].
            nn.Conv2d(71, 128, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 1, 1, 0, bias = False),
            
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        
        self._linear=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 9, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 235)
        )
        
        self._resnet=nn.Sequential(
            *(rescnn(128) for _ in range(4)),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            
    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        
        obs=self._tower(obs)
        obs=self._resnet(obs)
        action_logits=self._linear(obs)
        
        # Invalid actions are removed by adding a large negative log mask.
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask
