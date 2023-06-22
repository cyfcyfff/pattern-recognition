'''
数据预处理
'''


import os
import os.path

import numpy as np
from PIL import Image, ImageEnhance
import shutil
from torchvision import transforms as tfs
import tqdm
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
dst_size = 192


# 分集,dataset train : test = 3 : 1
def split():
    root = './sized/images_clfs'
    for label in os.listdir(root):

        imgs = os.listdir(root + '/' + label)
        os.makedirs(root + '/' + label + '/' + 'train', exist_ok=True)
        os.makedirs(root + '/' + label + '/' + 'test',  exist_ok=True)

        #抽样
        for i in range(len(imgs)):
            split = 'test' if i % 4 == 0 else 'train'
            print(label)
            # shutil.move(src, dst) 将src移动到dst
            shutil.move(root + '/' + label + '/' + imgs[i], root + '/' + label + '/' + split + '/' + imgs[i])


# 将tif图片格式转为png格式
def tif2png():
    dataset_root = "./dataset/images/test/Base31"
    save_path = dataset_root.replace('dataset', 'dataset_png')
    # os.makedirs(save_path, exist_ok=True)
    tif_list = os.listdir(dataset_root)

    for i in tqdm.tqdm(tif_list):
        if i[-4:] == '.png':
            continue
        image_path = os.path.join(dataset_root, i)

        image = Image.open(image_path)
        save_img_path = os.path.join(dataset_root, i[:-4] + '.png')
        image.save(save_img_path)
        os.remove(image_path)

'''
图片数据预处理
截一个正方形
进行缩放
'''
def data_size():
    dataset_root = './dataset/images'
    save_path = dataset_root.replace('dataset', 'sized')
    os.makedirs(save_path, exist_ok=True)

    for img_name in tqdm.tqdm(os.listdir(dataset_root)):
    # img_name = Base11....

        for img_name1 in (os.listdir(os.path.join(dataset_root, img_name))):
            dataset_root_1 = os.path.join(dataset_root, img_name)
            frame = cv2.imread(os.path.join(dataset_root_1, img_name1))
            shape = frame.shape
            # 截正方形
            y, x = shape[0], shape[1]
            pad = y
            x_center = int(x/2)
            x_start = x_center - int(pad/2)
            x_end = x_start + pad
            crop = frame[:, x_start:x_end,:]
            # 缩放
            crop = cv2.resize(crop, dsize=(192, 192))
            cv2.imwrite(os.path.join(save_path, img_name1), crop)


'''
结合视网膜病变程度和黄斑水肿风险等级，将数据集划分为10类
'''
def clsf():
    cladict = {
        '00':'0',
        '0' :'0',
        '10':'1',
        '11':'2',
        '12':'3',
        '20':'4',
        '21':'5',
        '22':'6',
        '30':'7',
        '31':'8',
        '32':'9',
    }

    root = './sized/images'
    split_root = './sized/images_clfs'

    # 建立10个目录
    for i in range(10):
        os.makedirs(os.path.join(split_root, str(i)), exist_ok=True)

    xls_path = './dataset/annotation of images/final.xls'
    xlsinfo = pd.read_excel(xls_path)

    for line in range(len(xlsinfo)):
        # 图片名称
        imgname = xlsinfo['Image name'][line]

        # 视网膜病变等级 [0, 1, 2, 3]
        Retinopathy_grade = xlsinfo['Retinopathy grade'][line]

        # 黄斑水肿风险 [0, 1, 2, 3]
        Risk_of_macular_edema = xlsinfo['Risk of macular edema '][line]

        stg = str(Retinopathy_grade) + str(Risk_of_macular_edema)
        split_num = cladict[stg]
        try:
            # png复制到split
            shutil.copy(os.path.join(root, imgname[:-4] + '.png'), os.path.join(split_root, split_num))
        except:
            print(imgname, 'error')


def balance_data(balance_target = 250):
    def img_strengthen(img):
        img = tfs.RandomOrder([
            tfs.RandomHorizontalFlip(),    # 按概率对图片进行水平翻转
            tfs.RandomCrop(192),           # 对图片进行剪裁
            tfs.RandomRotation(50),        # 随机旋转50°
        ])(img)

        return img

    root = './sized/images_clfs'
    for label in os.listdir(root):
        # len_data是当前文件夹中所有的样本数
        len_data = len(os.listdir(os.path.join(root, label)))
        temp = balance_target - len_data    # 250 - len_data
        if temp > 0:       #说明当前样本数小于250
            imgs = os.listdir(os.path.join(root, label))
            for img in imgs:

                # 打开图片
                rgb = Image.open(os.path.join(root, label, img))

                for i in range(temp // len_data):  #整除
                    name = img.split('.')[0] + f'str_{i}_' + '.jpg'
                    nimg = img_strengthen(rgb)
                    nimg.save(os.path.join(root, label, name))


def classes_cnt():
    root = './sized/images_clfs'
    x = []
    y = []
    for label in os.listdir(root):
        # len_data是当前文件夹中所有的样本数
        len_data = len(os.listdir(os.path.join(root, label)))
        #print(str(label) + ':' + str(len_data))
        x.append(str(label))
        y.append(str(len_data))
        print("label:" + str(label) + '----' + "data_num:" + str(len_data))
    x1 = np.array(x)
    y1 = np.array(y)

def arg_img():
    root = './sized/images_clfs'
    dst_root = 'data'
    splites = ['train', 'test']

    str_rate = 6
    def img_strengthen(img, mode):
        if mode == 'train':
            img = tfs.RandomOrder([
                tfs.RandomHorizontalFlip(),
                tfs.RandomCrop(192),
                tfs.RandomRotation(20),
                tfs.ColorJitter(brightness=0.5, contrast=0.5)
            ])(img)
        return img

    for label in os.listdir(root):
        if label == '1':
            str_rate = 7
        elif label == '0':
            str_rate = 2
        else:
            str_rate = 6

        for split in splites:
            # 图片读取地址
            dir = root + '/' + label + '/' + split

            # 图片存储地址
            dst_dir = dst_root + '/' + label + '/' +split
            os.makedirs(dst_dir, exist_ok=True)
            imgs = os.listdir(dir)
            for img in imgs:
                rgb = Image.open(dir + '/' + img)
                rgb.save(dst_dir + '/' + img)
                print(dir+'/'+img)
                for i in range(str_rate):
                    name = img.split('.')[0] + f'str_{i}_' + '.jpg'
                    nimg = img_strengthen(rgb, split)
                    nimg.save(dst_dir + '/' + name)




if __name__ == "__main__":
    arg_img()