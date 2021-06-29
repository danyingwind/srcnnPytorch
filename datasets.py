# HDF（Hierarchical Data Format）指一种为存储和处理大容量科学数据设计的文件格式及相应库文件。
# h5py是Python语言用来操作HDF5的模块
#import h5py
import glob
import io, os, random
import PIL.Image as pil_image
import numpy as np
from torch.utils.data import Dataset#用于实现数据读取的工具包
from torch.utils.data import DataLoader
from utils import convert_rgb_to_y


# class TrainDataset(Dataset):# 训练集的处理
#     def __init__(self, h5_file): # h5_file是指HDF文件
#         super(TrainDataset, self).__init__()# super()与调用父类的函数有关，表示采用父类的__init__()函数
#         self.h5_file = h5_file#初始化变量

#     def __getitem__(self, idx):
#         with h5py.File(self.h5_file, 'r') as f:#读取文件
#             # np.expand_dims(a, axis=0)表示在多维数组0维，也就是最外围，添加数据，数组(1,2,3)一共有0、1、2、3四个维度可以添加数据；0维度就是直接给原来数组外面加括号
#             # 我认为这里增加1维是为了当作数据集的标号？
#             return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)
#             # 返回的是1个元组，在python中[]为列表，{}为集合，()为元组，["":,]为字典

#     def __len__(self):
#         with h5py.File(self.h5_file, 'r') as f:
#             #返回的是元素的个数
#             return len(f['lr'])


# class EvalDataset(Dataset): # 验证集的处理
#     def __init__(self, h5_file):
#         super(EvalDataset, self).__init__()# super()与调用父类的函数有关，表示采用父类的__init__()函数
#         self.h5_file = h5_file#初始化变量

#     def __getitem__(self, idx):#这里和train不太一样
#         with h5py.File(self.h5_file, 'r') as f:
#             # 这里数据归一化的处理方法和TrainDataset意思相同，[:,:]的方法在python3中已经不使用了
#             return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

#     def __len__(self):
#         with h5py.File(self.h5_file, 'r') as f:
#             return len(f['lr']) # 返回元素个数

# 这里虽然设置的是读取训练集，但是按照lyp的说法，可以直接读入所有数据，然后使用random_split划分数据集
class TrainDataset2(Dataset):
    # 这里通常是构造所有数据的地址列表
    def __init__(self):
        self.hr_patches = [] # 这里hr对应原图
        self.lr_patches = [] # 这里lr对应压缩图，但是原图和压缩图的大小是一样的，只是来自不同的文件夹
        hr_images_dir = "/home/wangdanying/SRCNN/*****"
        lr_images_dir = "/home/wangdanying/SRCNN/*****"
        image_width = 1280
        image_height = 1280
        patch_size = 32 #使用SRCNN的初始配置为33
        stride = 16 #SRCNN的初始配置为14

        for image_path in sorted(glob.glob('{}/*'.format(hr_images_dir))):
            img = pil_image.open(image_path).convert('RGB')
            hr = img.crop([0,0,image_width,image_height])
            hr = np.array(hr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            # CNN处理图像的时候，不是一次输入一整块，而是把图像划分成多个patch
            # 获取每张图的hr_patches
            for i in range(0, hr.shape[0] - patch_size + 1, stride): 
                for j in range(0, hr.shape[1] - patch_size + 1, stride):
                    self.hr_patches.append(hr[i:i + patch_size, j:j + patch_size])

        for image_path in sorted(glob.glob('{}/*'.format(lr_images_dir))):
            img = pil_image.open(image_path).convert('RGB')
            lr = img.crop([0,0,image_width,image_height])
            lr = np.array(lr).astype(np.float32)
            lr = convert_rgb_to_y(lr)
            # CNN处理图像的时候，不是一次输入一整块，而是把图像划分成多个patch
            # 获取每张图的lr_patches
            for i in range(0, lr.shape[0] - patch_size + 1, stride): 
                for j in range(0, lr.shape[1] - patch_size + 1, stride):
                    self.lr_patches.append(lr[i:i + patch_size, j:j + patch_size])
            

        # 经过上述for循环，已经获得了所有图像分割后的patch，将patches转换成array
        self.num = len(self.lr_patches)
        self.lr_patches = np.array(self.lr_patches)
        self.hr_patches = np.array(self.hr_patches)

        ####################################注意要将接下来的数据放入字典
        # 是不是数据不放入字典也可以

        ###########################################################

        # self.num = len(self.lr_patches)
        # self.width = 1280
        # self.high = 1280 # 实际上高度是不确定的
        # self.images_list = []
        # self.num = 0
        
        # images_list = os.listdir(filepath);
        # self.num = len(self.images_list)
        # imgs = []
        # for i in range(self.num):
        #     imgs.append(Image.open(self.images_list[i]))
        
        # # 将图像数据转化成array，再转换成tensor
        # imgs = [torch.from_numpy((np.array(img).astype(np.float32)/255.)  for lrt in lrts]



    # 这里定义获取数据的方法
    def __getitem__(self, index):
        return np.expand_dims(self.lr_patches[index]/255., 0), np.expand_dims(self.hr_patches[index]/255., 0)

    

    # 这里返回数据的数量
    def __len__(self):
        return self.num
            
class EvalDataset2(Dataset):
    # 这里通常是构造所有数据的地址列表
    def __init__(self):
        # 注意这里的数据是字典索引下得到的列表
        self.hr_patches = {} # 这里hr对应原图
        self.lr_patches = {} # 这里lr对应压缩图，但是原图和压缩图的大小是一样的，只是来自不同的文件夹
        hr_images_dir = "/home/wangdanying/SRCNN/*****"
        lr_images_dir = "/home/wangdanying/SRCNN/*****"
        image_width = 1280
        image_height = 1280
        patch_size = 32 #使用SRCNN的初始配置为33
        stride = 16 #SRCNN的初始配置为14

        #############################################################################
        #这里需要注意，是不是一张图像对应一个验证集
        #############################################################################
        for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(hr_images_dir)))):
            img = pil_image.open(image_path).convert('RGB')
            hr = img.crop([0,0,image_width,image_height])
            hr = np.array(hr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            # CNN处理图像的时候，不是一次输入一整块，而是把图像划分成多个patch
            # 获取每张图的hr_patches
            for i in range(0, hr.shape[0] - patch_size + 1, stride): 
                for j in range(0, hr.shape[1] - patch_size + 1, stride):
                    self.hr_patches[str(i)].append(hr[i:i + patch_size, j:j + patch_size])

        for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(lr_images_dir)))):
            img = pil_image.open(image_path).convert('RGB')
            lr = img.crop([0,0,image_width,image_height])
            lr = np.array(lr).astype(np.float32)
            lr = convert_rgb_to_y(lr)
            # CNN处理图像的时候，不是一次输入一整块，而是把图像划分成多个patch
            # 获取每张图的lr_patches
            for i in range(0, lr.shape[0] - patch_size + 1, stride): 
                for j in range(0, lr.shape[1] - patch_size + 1, stride):
                    self.lr_patches[str(i)].append(lr[i:i + patch_size, j:j + patch_size])
            

        # 经过上述for循环，已经获得了所有图像分割后的patch，将patches转换成array
        self.num = len(self.lr_patches)
        self.lr_patches = np.array(self.lr_patches)
        self.hr_patches = np.array(self.hr_patches)

        ####################################注意要将接下来的数据放入字典
        # 是不是数据不放入字典也可以？

        ###########################################################

        # self.num = len(self.lr_patches)
        # self.width = 1280
        # self.high = 1280 # 实际上高度是不确定的
        # self.images_list = []
        # self.num = 0
        
        # images_list = os.listdir(filepath);
        # self.num = len(self.images_list)
        # imgs = []
        # for i in range(self.num):
        #     imgs.append(Image.open(self.images_list[i]))
        
        # # 将图像数据转化成array，再转换成tensor
        # imgs = [torch.from_numpy((np.array(img).astype(np.float32)/255.)  for lrt in lrts]



    # 这里定义获取数据的方法
    def __getitem__(self, index):
        return np.expand_dims(self.lr_patches[index]/255., 0), np.expand_dims(self.hr_patches[index]/255., 0)

    

    # 这里返回数据的数量
    def __len__(self):
        return self.num