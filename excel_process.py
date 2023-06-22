import os
import os.path
import tqdm
import pandas as pd
root = './dataset/annotation of images'
save_root = './dataset/annotation of images/final.xls'
img = []
for data_name in tqdm.tqdm(os.listdir(root)):
    inter_root = os.path.join(root, data_name)
    xlsinfo = pd.read_excel(inter_root)
    img.append(xlsinfo)

df_concat = pd.concat(img)
df_concat.to_excel(save_root, index=None)