import pickle
#转为DataSet类型
from torch.utils.data import Dataset


def read_data(tag):
    with open("./dataset/" + tag + '_set.pkl', 'rb') as fp:
        return pickle.load(fp)


def read_label(tag):
    with open("./dataset/" + tag + '_label.pkl', 'rb') as fp:
        return pickle.load(fp)


class TrainDataset(Dataset):
    def __init__(self):
        self.data = read_data('train')
        self.target = read_label('train')
        self.classes = ['positive', 'negative']

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self):
        self.data = read_data('test')
        self.target = read_label('test')
        self.classes = ['positive', 'negative']

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)