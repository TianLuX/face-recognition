# coding:utf-8
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
from module import ContrastiveLoss, VGG16
import torch.nn.functional as f

# 判断可用设备类型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

# our dataset
train_set = TrainDataset()
test_set = TestDataset()
# print('the size of scene aware train set {}'.format(len(train_set)))
# print('the size of scene aware test set {}'.format(len(test_set)))

# load data using DataLoader
batch_size = 32
train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_set, shuffle=True, batch_size=1)

#实例化模型
net = VGG16()
loss_function = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.00001)

net.to(device)
loss_function.to(device)

#训练部分
epochs = 100
loss_log = []
acc_log = []

for epoch in range(epochs):
    net.train()#使用dropout
    for step,data in enumerate(train_dataloader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        img0 = imgs[:, 0, :, :, :]
        img1 = imgs[:, 1, :, :, :]
        # print(labels.shape)
        # print(img0.shape)
        # print(img1.shape)
        optimizer.zero_grad()
        output = net(img0, img1)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print("Epoch number {}; Current loss {};".format(epoch, loss))
            loss_log.append(loss.item())

    #训练一个epoch保存loss和参数
    # torch.save(net.state_dict(), 'weights.pkl')
    # torch.save(loss_log, 'loss_log.pkl')

    # 测试部分
    acc = 0.0
    for inx, data in enumerate(test_dataloader):
        imgs, label = data
        imgs = imgs.to(device)
        label = label.to(device)
        img0 = imgs[:, 0, :, :, :]
        img1 = imgs[:, 1, :, :, :]
        # print(img0.shape)
        # print(img1.shape)
        output = net(img0, img1)
        tag = 1 if output >= 0.5 else 0
        if label == tag:
            acc += 1
    acc_log.append(acc / len(test_set))
    print("Epoch number {}; Current acc {};".format(epoch,acc / len(test_set)))
    # 训练一个epoch保存loss和参数
    # torch.save(acc_log, 'acc_log.pkl')




