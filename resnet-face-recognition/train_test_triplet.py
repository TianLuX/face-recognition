# coding:utf-8
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
from torch import optim, nn
from module import Resnet50,ContrastiveLoss,MyResnet50,Bottleneck
import torch
import torch.nn.functional as f

# 判断可用设备类型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

# our dataset
train_set = TrainDataset('train')
train_for_margin_set = TrainDataset('train_for_margin')
test_set = TestDataset()

# load data using DataLoader
batch_size = 32
train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
train_for_margin_dataloader = DataLoader(train_for_margin_set, shuffle=True, batch_size=1)
test_dataloader = DataLoader(test_set, shuffle=True, batch_size=1)

# 实例化模型
# net = MyResnet50(Bottleneck)
net = Resnet50()
loss_function = nn.TripletMarginLoss(margin=1.0)#使用自带三元损失函数，也可以用module里写的
loss_function2 = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.00001)
# optimizer = optim.SGD(net.parameters(), lr=0.00001)
net.to(device)
loss_function.to(device)
loss_function2.to(device)

# 训练部分
epochs = 10
# 用于保存loss和acc
loss_log = []
acc_log = []

for epoch in range(epochs):
    # 使用第一个训练集进行训练
    net.train()
    for step, data in enumerate(train_dataloader):

        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        img0 = imgs[:, 0, :, :, :]
        img1 = imgs[:, 1, :, :, :]
        img2 = imgs[:, 2, :, :, :]

        optimizer.zero_grad()
        anchor = net(img0)
        positive = net(img1)
        negative = net(img2)

        loss = loss_function(anchor, positive, negative)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print("Epoch number {}; Current loss {};".format(epoch, loss))
            loss_log.append(loss.item())

    # # 每一个epoch保存权重和loss
    torch.save(net.state_dict(), 'weights.pkl')
    # torch.save(loss_log, 'loss_log.pkl')

    # 使用第二个训练集进行训练
    # 并且确定margin
    positive_output = []
    negative_output = []
    for step, data in enumerate(train_for_margin_dataloader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        img0 = imgs[:, 0, :, :, :]
        img1 = imgs[:, 1, :, :, :]

        output1 = net(img0)
        output2 = net(img1)

        loss2 = loss_function2(output1, output2, labels)
        loss2.backward()
        optimizer.step()

        distance = f.pairwise_distance(output1, output2, keepdim=True)

        positive_output.append(distance) if labels == 1 else negative_output.append(distance)

    positive_output = sorted(positive_output)
    negative_output = sorted(negative_output)
    # print("positive_output:{}".format(positive_output))
    # print("negative_output:{}".format(negative_output))

    # 数组中位数下标
    median_positive_index = int(len(positive_output) / 2)
    median_negative_index = int(len(negative_output) / 2)
    # print('positive', positive_output[median_positive_index])
    # print('negative', negative_output[median_negative_index])
    margin = (positive_output[median_positive_index] + negative_output[median_negative_index]) / 2
    # print('margin', margin)

    if epoch % 5 == 0:
        # 每五个epoch测试一次
        with torch.no_grad():
            # 测试
            acc = 0.0
            for inx, data in enumerate(test_dataloader):
                imgs, label = data
                imgs = imgs.to(device)
                label = label.to(device)
                img0 = imgs[:, 0, :, :, :]
                img1 = imgs[:, 1, :, :, :]
                output1 = net(img0)
                output2 = net(img1)
                distance = f.pairwise_distance(output1, output2, keepdim=True)
                tag = 1 if distance < margin else 0
                if label == tag:
                    acc += 1
            acc_log.append(acc / len(test_set))
            print("Epoch number {}; Current acc {};".format(epoch, acc / len(test_set)))
