from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy import interpolate
from sklearn.metrics import auc, roc_curve

from torch.nn import functional as F
from torchvision import  transforms
from tqdm import tqdm

from data import MyDataSet
from lenet import device

to_pil_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

net = torch.load('model_S.pth', map_location=torch.device(device))
net.eval()

dataset_test = MyDataSet(split='test')
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=0)
def one_hot_to_label(one_hot):
    return torch.argmax(one_hot, dim=-1).cpu().item()



def run_one(cv_rgb):
    tensor = to_tensor(cv_rgb).to(device)
    x = tensor.unsqueeze(0)
    y1, y2 = net(x)
    y1 = one_hot_to_label(y1)
    y2 = one_hot_to_label(y2)

    return y1, y2


if __name__ == '__main__':
    with torch.no_grad():
        y_score1 = None
        y_label1 = None
        y_score2 = None
        y_label2 = None
        num = 0.0
        right_num = 0.0
        for batch_id, data in tqdm(enumerate(dataloader_test, 0), total=len(dataloader_test), smoothing=0.9):
            rgb, label1, label2 = data
            y_score1, y_score2 = net(rgb)
            y1 = one_hot_to_label(y_score1)
            y2 = one_hot_to_label(y_score2)
            Y1 = y_score1.detach().numpy()
            Y2 = y_score2.detach().numpy()
            if y2 == label2:
                right_num += 1

            num += 1

        print('acc=' + str(right_num/num))


















