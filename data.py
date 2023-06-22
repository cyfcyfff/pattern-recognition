from torch.utils.data import Dataset
from torchvision import transforms as tfs
import os
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import cv2
import torch
#from lenet import device
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
root = './data'
cache = True
cache_pool = {}

label_d = {
    '0': [0, 0],
    '1': [1, 0],
    '2': [1, 1],
    '3': [1, 2],
    '4': [2, 0],
    '5': [2, 1],
    '6': [2, 2],
    '7': [3, 0],
    '8': [3, 1],
    '9': [3, 2]
}

data_strengthen_rate = 49

img_strengthen = tfs.Compose([
    tfs.RandomHorizontalFlip(),
    tfs.RandomCrop(192),
    tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
])


class MyDataSet(Dataset):
    def __init__(self, split='train'):
        self.root = root
        self.split = split
        self.data = []
        self.label1 = []
        self.label2 = []

        self.to_tensor = transforms.ToTensor()

        for label in os.listdir(root):
            dir = root + '/' + label + '/' + split
            imgs = os.listdir(dir)
            for img in imgs:
                self.data.append(dir + '/' + img)
                self.label1.append(label_d[label][0])
                self.label2.append(label_d[label][1])

    def __len__(self):
        return len(self.data)

    def gen_item(self, index):
        rgb = self.data[index]
        rgb = cv2.imread(rgb)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        tensor = self.to_tensor(rgb).to(device)
        label1 = int(self.label1[index])
        label2 = int(self.label2[index])
        return tensor, label1, label2

    def _get_item(self, index):
        if not cache:
            return self.gen_item(index)
        if index in cache_pool:
            return cache_pool[index]
        item = self.gen_item(index)
        cache_pool[index] = item
        return item

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    a = MyDataSet('train')
    #rgb, (l1, l2, l3) = a[0]

    #to_pil_image(rgb.to('cpu')).save('data0.jpg')
    #print(l1, l2, l3)
