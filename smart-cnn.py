from torch.functional import Tensor
import torch.nn as nn
import torch
from torch.nn.modules.dropout import Dropout
from torch.nn import functional as F
import GENet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = GENet.genet_small(pretrained=True).to(device)


class MyNet(nn.Module):
    def __init__(self, class_nums_1 = 4, class_nums_2 = 3):
        super(MyNet, self).__init__()
        self.backbone = backbone
        self.classfiler1 = nn.Sequential(
                            nn.Linear(1000 ,256),   
                            nn.ReLU(),
                            Dropout(0.4),
                            nn.Linear(256, class_nums_1)
                        )
        self.classfiler2 = nn.Sequential(
                            nn.Linear(1000 ,256),   
                            nn.ReLU(),
                            Dropout(0.4),
                            nn.Linear(256, class_nums_2)
                        )
    
    def forward(self, x: Tensor):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        c1 = self.classfiler1(x)
        c2 = self.classfiler2(x)
        return c1, c2