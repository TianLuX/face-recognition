# coding:utf-8
import pickle
import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn.functional as f



# 读入训练数据和测试数据
def readFile(path):
    with open(path) as f:
        datas = f.readlines()
        data_count = datas[0]
        data = []
        for t_data in datas[1:]:
            t_data = t_data.strip('\n')  # 去掉换行符
            t_data = t_data.split('\t')  # 按照tab键进行分割
            data.append(t_data)  # 训练数据集

    return int(data_count), data



# 将图片转为对应的张量
def jpg_to_tensor(datas):

    tensor_datas = []
    label = []
    for data in datas:
        data = np.array(data)
        img_path1 = "lfw_align/" + data[0] + "/" + data[0] + "_" + str(data[1]).zfill(4) + ".jpg"
        # 同一张人脸
        if data.size == 3:
            img_path2 = "lfw_align/" + data[0] + "/" + data[0] + "_" + str(data[2]).zfill(4) + ".jpg"
            bool = True
        else:  # 两张人脸
            img_path2 = "lfw_align/" + data[2] + "/" + data[2] + "_" + str(data[3]).zfill(4) + ".jpg"
            bool = False

        # 可能有图片在align环节不成功
        img = np.array(plt.imread(img_path1))
        img = cv2.resize(img,(224,224),interpolation=cv2.INTER_LINEAR)/255
        height, width, channel = img.shape
        img0 = np.zeros((channel, height, width), dtype=np.float32)
        img0[0, :, :] = img[:, :, 0]
        img0[1, :, :] = img[:, :, 1]
        img0[2, :, :] = img[:, :, 2]


        img = np.array(plt.imread(img_path2))
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR) / 255
        img1 = np.zeros((channel, height, width), dtype=np.float32)
        img1[0,:,:] = img[:, :, 0]
        img1[1, :, :] = img[:, :, 1]
        img1[2, :, :] = img[:, :, 2]


        tensor_datas.append([img0, img1])
        if bool:
            label.append(1)
        else:
            label.append(0)
    return np.array(tensor_datas), np.array(label)




# 获取训练数据和测试数据
train_path = "pairsDevTrain.txt"
train_data_count, train_data = readFile(train_path)
train_data, train_label = jpg_to_tensor(train_data)
# 形状为2200*2*224*224*3，小于等于1100的是一个人，大于1100的是两个人
# print(train_data.shape)
# print(train_label.shape)

# #测试数据
test_path = "pairsDevTest.txt"
test_data_count, test_data = readFile(test_path)
test_data, test_label = jpg_to_tensor(test_data)
# print(test_data.shape)
# print(test_label.shape)


# 处理完数据存入pkl
with open('dataset/train_set.pkl', 'wb') as fp:
    pickle.dump(train_data.astype(np.float32), fp)
with open('dataset/train_label.pkl', 'wb') as fp:
    pickle.dump(train_label.astype(np.float32), fp)
with open('dataset/test_set.pkl', 'wb') as fp:
    pickle.dump(test_data.astype(np.float32), fp)
with open('dataset/test_label.pkl', 'wb') as fp:
    pickle.dump(test_label.astype(np.float32), fp)
