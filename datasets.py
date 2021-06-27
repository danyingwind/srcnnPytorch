# HDF（Hierarchical Data Format）指一种为存储和处理大容量科学数据设计的文件格式及相应库文件。
# h5py是Python语言用来操作HDF5的模块
import h5py
import io, os, random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset#用于实现数据读取的工具包
from torch.utils.data import DataLoader


class TrainDataset(Dataset):# 训练集的处理
    def __init__(self, h5_file): # h5_file是指HDF文件
        super(TrainDataset, self).__init__()# super()与调用父类的函数有关，表示采用父类的__init__()函数
        self.h5_file = h5_file#初始化变量

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:#读取文件
            # np.expand_dims(a, axis=0)表示在多维数组0维，也就是最外围，添加数据，数组(1,2,3)一共有0、1、2、3四个维度可以添加数据；0维度就是直接给原来数组外面加括号
            # 我认为这里增加1维是为了当作数据集的标号？
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)
            # 返回的是1个元组，在python中[]为列表，{}为集合，()为元组，["":,]为字典

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            #返回的是元素的个数
            return len(f['lr'])


class EvalDataset(Dataset): # 验证集的处理
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()# super()与调用父类的函数有关，表示采用父类的__init__()函数
        self.h5_file = h5_file#初始化变量

    def __getitem__(self, idx):#这里和train不太一样
        with h5py.File(self.h5_file, 'r') as f:
            # 这里数据归一化的处理方法和TrainDataset意思相同，[:,:]的方法在python3中已经不使用了
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr']) # 返回元素个数

# 这里虽然设置的是读取训练集，但是按照lyp的说法，可以直接读入所有数据，然后使用random_split划分数据集
class TrainDataset2(Dataset):
    # 这里通常是构造所有数据的地址列表
    def __init__(self):
        self.width = 1280
        self.high = 1280 # 实际上高度是不确定的
        self.images_list = []
        self.num = 0
        file_path = "/home/wangdanying/SRCNN/R1TexturePng"
        images_list = os.listdir(filepath);
        self.num = len(self.images_list)
        imgs = []
        for i in range(self.num):
            imgs.append(Image.open(self.images_list[i]))
        
        # 将图像数据转化成array，再转换成tensor
        imgs = [torch.from_numpy((np.array(img).astype(np.float32)/255.)  for lrt in lrts]



    # 这里定义获取数据的方法
    def __getitem__(self, index):

        return self.images_list[index]

    

    # 这里返回数据的数量
    def __len__(self):
        return self.num
            
