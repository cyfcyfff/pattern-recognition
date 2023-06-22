import torch
from torch.nn import functional as F
from tqdm import tqdm
import datetime

import lenet
from data import MyDataSet
from lenet import device
from torch.utils.tensorboard import SummaryWriter


def one_hot_to_label(one_hot):
    return torch.argmax(one_hot, dim=-1).cpu().item()


def validate(net, data_loader):
    with torch.no_grad():
        net.eval()
        corrs = 0
        for rgb, label1, label2 in data_loader:
            y1, y2 = net(rgb)
            for sample_id in range(label1.shape[0]):
                if label1[sample_id].item() == one_hot_to_label(y1[sample_id]) and label2[sample_id].item() == one_hot_to_label(y2[sample_id]):
                    corrs += 1
        net.train()
        return [corrs / len(data_loader.dataset)]


def train(net, dataloader_train, dataloader_test, epochs, optimizer, scheduler=None, logger=None):
    n_iters = 0

    for epoch in range(epochs):
        for batch_id, data in tqdm(enumerate(dataloader_train, 0), total=len(dataloader_train), smoothing=0.9):
            rgb, label1, label2 = data
            optimizer.zero_grad()
            net = net.train()
            y1, y2 = net(rgb)
            pred1 = F.log_softmax(y1, -1)
            pred2 = F.log_softmax(y2, -1)

            loss = F.nll_loss(pred1, label1.to(device))
            loss += F.nll_loss(pred2, label2.to(device))
            loss.backward()
            optimizer.step()
            if logger is not None:
                loss_group = 'Loss_class'
                loss_name = 'NLL'
                logger.add_scalar(f'train/{loss_name}', loss, n_iters)
            n_iters += 1
        training_accs = validate(net, dataloader_train)
        testing_accs = validate(net, dataloader_test)

        tqdm.write(f'{training_accs=}  {testing_accs=} loss = {loss.item()}')
        torch.save(net, 'model.pth')
        if scheduler is not None:
            scheduler.step()


if __name__ == '__main__':
    torch.cuda.set_per_process_memory_fraction(0.8, 0)

    # =========== hyper params =============
    batch_size = 1
    epoches = 100
    learning_rate = 0.001
    # ============== End ===================
    out_path = 'experiment'
    experiment_name = 'exp1'
    run_info = ''
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    infostamp = f'_{run_info.strip()}' if run_info.strip() else ''
    logger = SummaryWriter(f'{out_path}/runs/{timestamp}{infostamp}')

    net = lenet.LeNet().to(device)
    dataset_train = MyDataSet(split='train')
    dataset_test = MyDataSet(split='test')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    # optimizer_finetune = torch.optim.SGD(net.fc.parameters(), learning_rate)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)  # for Adam
    train(net, dataloader_train, dataloader_test, epoches, optimizer, scheduler, logger)