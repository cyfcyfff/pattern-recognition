import pandas as pd

path_r = r"..\new\data1.txt"
with open(path_r, "r") as f:  # 打开文件
    data = f.read()  # 读取文件
path_w = r"C:..\now\1.txt"

epo = []
tra = []
tes = []
los = []
tim = []

with open(path_w, "w") as f:    # 以写的方式打开结果文件
    # training_accs
    e = 1 # epoch数
    i = -1  # 用来记录本次循环数据所在位置
    j = data.rfind('training_accs=')   # 找到最后一条数据所在的位置
    while i < j:
        start = i+1  # 每次查找都要从上一条数据之后的位置开始，默认从0开始
        i = data.index('training_accs=', start)    # 从start开始查找，返回training_accs第一次出现的位置
        result = data[i+14:i+14+18]     # i+14是从training_accs的位置往后移，也就是说去掉“training_accs=”，i+14+18是指数值部分取18位
        tra.append(result)
        epo.append(e)
        e += 1
    # testing_accs
    i = -1  # 用来记录本次循环数据所在位置
    j = data.rfind('testing_accs=')  # 找到最后一条数据所在的位置
    while i < j:
        start = i + 1  # 每次查找都要从上一条数据之后的位置开始，默认从0开始
        i = data.index('testing_accs=', start)  # 从start开始查找，返回testing_accs=第一次出现的位置
        result = data[i + 13:i + 13 + 18]  # i+13是从testing_accs=的位置往后移，也就是说去掉“testing_accs=”，i+13+18是指数值部分取18位
        tes.append(result)
    # loss
    i = -1  # 用来记录本次循环数据所在位置
    j = data.rfind('loss = ')  # 找到最后一条数据所在的位置
    while i < j:
        start = i + 1  # 每次查找都要从上一条数据之后的位置开始，默认从0开始
        i = data.index('loss = ', start)  # 从start开始查找，返回loss=第一次出现的位置
        result = data[i + 7:i + 7 + 17]  # i+14是从loss = 的位置往后移，也就是说去掉“loss=”，i+14+17是指数值部分取17位
        los.append(result)
    # time
    i = -1  # 用来记录本次循环数据所在位置
    j = data.rfind('172/172')  # 找到最后一条数据所在的位置
    while i < j:
        start = i + 1  # 每次查找都要从上一条数据之后的位置开始，默认从0开始
        i = data.index('172/172 ', start)  # 从start开始查找，返回172/172第一次出现的位置
        result = data[i + 8:i + 8 + 5]  # i+8是从172/172的位置往后移，也就是说去掉“172/172”，i+8+5是指数值部分取17位
        los.append(result)

da = {'epoch': epo, 'training_accs': tra, 'testing_accs=': tes, 'loss=': los, 'time': tim}
df = pd.DataFrame(da)
df.to_excel('resul.xlsx', index=False)