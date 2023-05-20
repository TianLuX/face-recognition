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

# 同一个人
def sampling_same():
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

# 两个人
def sampling_diff():
    t = random.randint(0, 2)#用于判断是从有多个照片的人中选择还是只有一张照片的人中选择
    if t == 0:
        name_list = random.sample(names_one.items(), 2)  # 随机取两个名字
        name1 = name_list[0][0]
        name2 = name_list[1][0]
        id1 = 1
        id2 = 1
    elif t == 1:
        name_list = random.sample(names_one.items(), 1)  # 随机取一个名字
        name1 = name_list[0][0]
        id1 = 1
        name_list = random.sample(names_more.items(), 1)  # 随机取一个名字
        name2 = name_list[0][0]
        id2 = random.randint(1, int(name_list[0][1]))
    else:
        name_list = random.sample(names_more.items(), 2) # 随机取两个名字
        name1 = name_list[0][0]
        id1 = random.randint(1, int(name_list[0][1]))
        name2 = name_list[1][0]
        id2 = random.randint(1, int(name_list[1][1]))

    return name1,id1,name2,id2

def addSampleFile(path, name_count):
    '''
        补充数据集
        :param path: 存放文件的路径
        :param name_count: 样本数量
        :return:
    '''
    with open(path) as f:
        lines = f.readlines()
        lines[0] = lines[0].strip('\n')
        #在最后一行添加\n
        last = int(lines[0])*2
        s = lines[last]
        lines[last] = s + '\n'
        #更改数量
        t = int(lines[0]) + name_count
        lines[0] = str(t) + '\n'
        for i in range(name_count):
            name, id1, id2 = sampling_same()
            lines.append(str(name) + "\t" + str(id1) + "\t" + str(id2) + "\n")
        for i in range(name_count):
            name1, id1, name2, id2 = sampling_diff()
            if i == name_count - 1:
                lines.append(
                    str(name1) + "\t" + str(id1) + "\t" + str(name2) + "\t" + str(id2))
            else:
                lines.append(
                    str(name1) + "\t" + str(id1) + "\t" + str(name2) + "\t" + str(id2) + "\n")

    with open(path, "w") as f:
        for line in lines:
            f.write(line)


readNames("peopleDevTrain.txt")
path = "pairsDevTrain.txt"
name_count = 900
addSampleFile(path, name_count)