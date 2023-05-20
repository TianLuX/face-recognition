# coding:utf-8
# 从peopleDevTrain.txt中进行随机采样训练
import random
names_one = {}# 存放只有一张照片的人
names_more = {}# 存放多张照片的人

def readNames(path):
    '''
    读取所有的名称和对应的照片数量
    :param path: 文件路径，从peopleDevTrain.txt文件中读取
    :return: 返回人数
    '''
    with open(path) as f:
        lines = f.readlines()
        name_count = lines[0]
        for line in lines[1:]:
            line = line.strip('\n')  # 去掉换行符
            line = line.split('\t')  # 按照tab键进行分割
            if int(line[1]) == 1:
                names_one[line[0]] = int(line[1])
            else:
                names_more[line[0]] = int(line[1])
    return int(name_count)

# 随机采样anchor和positive
def sampling_positive():
    t = random.sample(names_more.items(), 1)  # 在有多张照片的人中随机取一个名字
    name = t[0][0]
    if int(t[0][1]) == 2:
        id1 = 1
        id2 = 2
    else:
        ID_list = random.sample(range(1, int(t[0][1])), 2)  # 在1和t[0][1]之间生成两个不同随机数，为照片编号
        id1 = ID_list[0]
        id2 = ID_list[1]
    return name, id1, id2

# 随机采样negative
def sampling_negative():
    t = random.randint(0, 1)  #用于判断是从有多个照片的人中选择还是只有一张照片的人中选择
    if t == 0:
        name_list = random.sample(names_one.items(), 1)  # 随机取一个名字作为负样本
        name = name_list[0][0]
        id = 1
    else:
        name_list = random.sample(names_more.items(), 1)  # 随机取一个名字负样本
        name = name_list[0][0]
        id = random.randint(1, int(name_list[0][1]))
    return name, id

def contrastFile(path, name_count):
    '''
    生成对比训练所需要的数据
    :param path: 存放文件的路径
    :param name_count: 样本数量
    :return:
    '''
    with open(path) as f:
        lines = f.readlines()
        lines.append(str(name_count) + "\n")
        for i in range(name_count):
            name1, id1, id2 = sampling_positive()

            while(True):
                name2, id3 = sampling_negative()
                if name1 != name2:#如果positive和negative名称一样，重新生成
                    break

            if i == name_count - 1:
                lines.append(
                    str(name1) + "\t" + str(id1) + "\t" + str(id2) + "\t" + str(name2) + "\t" + str(id3))
            else:
                lines.append(
                    str(name1) + "\t" + str(id1) + "\t" + str(id2) + "\t" + str(name2) + "\t" + str(id3) + "\n")

    with open(path, "w") as f:
        for line in lines:
            f.write(line)

readNames("peopleDevTrain.txt")
# print(len(names_one))
# print(names_one)
# print(len(names_more))
# print(names_more)
path = "contrastTrain.txt"
name_count = 3000
contrastFile(path, name_count)