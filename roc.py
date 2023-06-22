from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import interp
from sklearn.metrics import auc, roc_curve
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from data import MyDataSet

# 注意data.py 中的device要和这里的对应
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

net = torch.load('model_Adamax.pth')


to_pil_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

dataset_test = MyDataSet(split='test')
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=0)
y_score1= None
y_label1 = None
y_score2= None
y_label2 = None
# 标签独热
def one_hot_to_label(one_hot):
    return torch.argmax(one_hot, dim=-1).cpu().item()

# 封装成端到端的函数，便于多线程部署到嵌入式设备
def run_one(cv_rgb):
    tensor = to_tensor(cv_rgb).to(device)
    x = tensor.unsqueeze(0)
    y1, y2 = net(x)
    y1 = one_hot_to_label(y1)
    y2 = one_hot_to_label(y2)
    return y1, y2

# 绘制ROC曲线
def get_roc(n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label1[:, i], y_score1[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label1.ravel(), y_score1.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    with torch.no_grad():
        y_score1 = None
        y_label1 = None
        y_score2 = None
        y_label2 = None
        for batch_id, data in tqdm(enumerate(dataloader_test, 0), total=len(dataloader_test), smoothing=0.9):
            rgb, label1, label2 = data
            if batch_id == 0:
                y_score1, y_score2 = net(rgb)
                y_label1 = torch.zeros(64, 4).scatter_(1, label1.unsqueeze(1), 1).cpu()
                y_label2 = torch.zeros(64, 3).scatter_(1, label2.unsqueeze(1), 1).cpu()
            else:
                temp1, temp2 = net(rgb)
                y_score1 = torch.cat((y_score1, temp1), dim=0)
                y_score2 = torch.cat((y_score2, temp2), dim=0)
                y_label1 = torch.cat((y_label1, torch.zeros(64, 4).scatter_(1, label1.unsqueeze(1), 1).cpu()), dim=0)
                y_label2 = torch.cat((y_label2, torch.zeros(64, 3).scatter_(1, label2.unsqueeze(1), 1).cpu()), dim=0)

                pass
        y_label1 = y_label1[:y_score1.shape[0]].numpy()
        y_label2 = y_label2[:y_score2.shape[0]].numpy()

        y_score1 = y_score1.cpu().detach().numpy()
        y_score2 = y_score2.cpu().detach().numpy()

        n_classes1 = 4
        n_classes2 = 3

        get_roc(n_classes1)
        get_roc(n_classes2)