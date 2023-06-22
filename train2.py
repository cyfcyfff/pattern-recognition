import torch
from torch.nn import functional as F
from tqdm import tqdm
import datetime

import lenet
from data import MyDataSet
from lenet import device
from torch.utils.tensorboard import SummaryWriter


def one_hot_to_label(one_hot):
    '''
    独热编码转成标签
    '''
    return torch.argmax(one_hot, dim=-1).cpu().item()   # 在one_hot张量的最后一个维度（概率）上寻找每个样本中概率最大的类别，并返回一个形状为(batch_size,)的张量


def validate(net, data_loader):
    '''
    在验证集上测试训练好的模型的性能
    :param net:模型
    :param data_loader:数据加载器
    :return:正确率
    '''
    with torch.no_grad():   # 禁用梯度计算
        net.eval()          # 切换为评估模式
        corrs = 0           # 记录预测正确的样本数量
        for rgb, label1, label2 in data_loader:
            y1, y2 = net(rgb)   # 得到模型输出的标签
            # 记录预测正确的样本数量：
            for sample_id in range(label1.shape[0]):
                if label1[sample_id].item() == one_hot_to_label(y1[sample_id]) and label2[sample_id].item() == one_hot_to_label(y2[sample_id]):
                    corrs += 1
        net.train()     # 切换回训练模式
        return [corrs / len(data_loader.dataset)]


def train(net, dataloader_train, dataloader_test, epochs, optimizer, scheduler=None, logger=None):
    '''
     训练函数
     :param net: PyTorch模型
     :param dataloader_train: 训练集数据加载器
     :param dataloader_test: 测试集数据加载器
     :param optimizer: 优化器
     :param epochs: 训练轮数
     :param scheduler: 学习率调度器
     :param logger: 日志记录器
     :return: 无返回值
    '''

    n_iters = 0  # 迭代次数
    for epoch in range(epochs):
        # 给dataloader_train里的batch整上编号；total：迭代器的总长度；0.9的进度条的平滑度
        for batch_id, data in tqdm(enumerate(dataloader_train, 0), total=len(dataloader_train), smoothing=0.9):
            rgb, label1, label2 = data      # 取出数据
            optimizer.zero_grad()           # 清空模型的所有参数的梯度
            net = net.train()               # 切换到训练模式
            y1, y2 = net(rgb)               # 得到预测结果
            pred1 = F.log_softmax(y1, -1)   # 转概率分布1
            pred2 = F.log_softmax(y2, -1)   # 转概率分布2

            loss = F.nll_loss(pred1, label1.to(device))     # 计算交叉熵损失1
            loss += F.nll_loss(pred2, label2.to(device))    # 计算交叉熵损失2
            loss.backward()     # 计算参数梯度
            optimizer.step()    # 根据模型参数的梯度更新模型参数
            if logger is not None:
                loss_group = 'Loss_class'                               # 定义损失组，用于在可视化工具中对损失进行分类和比较
                loss_name = 'NLL'                                       # 定义损失名称，用于在可视化工具中对损失进行分类和比较
                logger.add_scalar(f'train/{loss_name}', loss, n_iters)  # 使用logger记录损失值，以便于后续分析和可视化
            n_iters += 1
        training_accs = validate(net, dataloader_train)     # 计算准确率1
        testing_accs = validate(net, dataloader_test)       # 计算准确率2

        tqdm.write(f'epoch = {epoch} {training_accs=}  {testing_accs=} loss = {loss.item()}')   # 命令行界面输出结果
        torch.save(net, 'model.pth')    # 保存神经网络模型
        if scheduler is not None:
            scheduler.step()    # 更新学习率


if __name__ == '__main__':
    torch.cuda.set_per_process_memory_fraction(0.8, 0)

    # =========== hyper params =============
    batch_size = 64         # 每个批次中包含的样本数
    epoches = 200             # 模型训练的轮数
    learning_rate = 0.001   # 模型在训练过程中的学习速率
    # ============== End ===================
    out_path = 'experiment'     # 输出路径
    experiment_name = 'exp1'    # 实验名称
    run_info = ''               # 实验信息（可选）
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')   # 当前时间戳
    infostamp = f'_{run_info.strip()}' if run_info.strip() else ''      # 实验信息时间戳（可选）
    logger = SummaryWriter(f'{out_path}/runs/{timestamp}{infostamp}')   # 日志记录器对象

    net = lenet.LeNet().to(device)              # LeNet模型，将其移动到指定的设备（CPU或GPU）上
    dataset_train = MyDataSet(split='train')    # 训练数据集
    dataset_test = MyDataSet(split='test')      # 测试数据集
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    # optimizer_finetune = torch.optim.SGD(net.fc.parameters(), learning_rate)  # 优化全连接层的参数
    '''
    # Adam算法进行参数优化

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08
    )

    '''
    # Adagrad算法进行参数选择
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
    )
    '''
        图书馆赶人了，不找了，等明天：
    
        随机梯度下降（SGD）：是一种最基本的优化算法，每次更新模型时，都使用当前样本的梯度来更新模型参数。

        动量梯度下降（Momentum）：是在SGD算法的基础上引入动量（momentum）的一种优化算法，可以在梯度方向一致时加速收敛。

        Adagrad：是一种自适应学习率的优化算法，可以根据参数的历史梯度大小来动态调整学习率，从而不同参数可以使用不同的学习率。

        Adadelta：是一种自适应学习率的优化算法，可以根据参数的历史梯度和更新幅度来动态调整学习率，从而可以自适应地调整学习率大小。

        RMSprop：是一种自适应学习率的优化算法，可以根据参数的历史梯度平方的指数加权平均数来动态调整学习率，从而可以自适应地调整学习率大小。

        Adamax：是Adam优化算法的一种变体，可以在处理稀疏梯度时更加稳定和高效。

        SparseAdam：是Adam优化算法的一种变体，可以在处理稀疏梯度时更加稳定和高效。

        LBFGS：是一种基于拟牛顿法的优化算法，可以使用二阶信息来更新模型参数，从而加速收敛速度。
        
        '''

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)  # for Adam
    train(net, dataloader_train, dataloader_test, epoches, optimizer, scheduler, logger)    # 训练