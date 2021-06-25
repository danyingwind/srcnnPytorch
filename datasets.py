# HDF（Hierarchical Data Format）指一种为存储和处理大容量科学数据设计的文件格式及相应库文件。
# h5py是Python语言用来操作HDF5的模块
import h5py
import numpy as np
from torch.utils.data import Dataset#用于实现数据读取的工具包


class TrainDataset(Dataset):
    def __init__(self, h5_file): # h5_file是指HDF文件
        super(TrainDataset, self).__init__()# super()与调用父类的函数有关，表示采用父类的__init__()函数
        self.h5_file = h5_file#初始化变量

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:#读取文件
            #np.expand_dims(a, axis=0)表示在0位置添加数据，数组(1,2,3)一共有0、1、2、3四个位置可以添加数据。
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()# super()与调用父类的函数有关，表示采用父类的__init__()函数
        self.h5_file = h5_file#初始化变量

    def __getitem__(self, idx):#这里和train不太一样
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
