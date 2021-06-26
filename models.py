from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        #__init__()方法是一种特殊的方法，被称为类的构造函数或初始化方法
        #self代表类的实例而非类
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)#python中的//表示整数除法
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        # https://blog.csdn.net/qq_38863413/article/details/104108808
        # Conv2d(in_channels, out_channels, kernel_size, 
        #       stride=1,padding=0, dilation=1, groups=1,
        #       bias=True, padding_mode=‘zeros’)
        # in_channels指输入的数据通道数量，比如RGB图像就是3通道，由于RGB图像是三通道，那么对应的卷积核也应当是3通道，
        # 也就是一个卷积核有3个滤波器，分别对应3通道
        # out_channels对应卷积核的数量，无论卷积核的通道数为多少，1个卷积核就是对应1个out_channel
        # kernel_size只给一个数值的时候表示卷积核是正方形
        # padding是指在所有边界增加值为padding的大小


        self.relu = nn.ReLU(inplace=True)
        # https://blog.csdn.net/jzwong/article/details/103431786
        # inplace=true会使用ReLU输出的值更新输入，默认inplace=false即不更新输入

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x) #自定义函数，见本文件开头
        return x
