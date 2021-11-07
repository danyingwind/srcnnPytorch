from torch import nn
import commonEDSR as common
import torch.nn as nn


# 这里需要args，还需要common.py提供的default_conv函数
class srcnnEDSR(nn.Module):
    # 定义模型的各个组件==
    def __init__(self, args, conv=common.default_conv):
        #__init__()方法是一种特殊的方法，被称为类的构造函数或初始化方法
        super(srcnnEDSR, self).__init__()
        n_resblocks = args.n_resblocks # resblock的数量
        n_feats = args.n_feats # feature map的数量
        kernel_size = 3  # 卷积核的大小
        scale = args.scale[0] # 来自args的参数，wangdanying这里应当改为1
        act = nn.ReLU(True) 
        self.sub_mean = common.MeanShift(args.rgb_range) # RGB的最大值
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)



        #==========================================开始定义模型==========================================#

        # define head module
        # 模型的开头部分，1个卷积模块
        m_head = [conv(args.n_colors, n_feats, kernel_size)] #颜色通道数，特征图数量，卷积核尺寸

        # define body module
        # 模型的中间部分，n个残差模块
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        # 模型的中间部分，1个卷积模块
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        #模型的结尾部分，上采样模块+卷积模块
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        # 这里是给各个模块进行重命名
        self.head = nn.Sequential(*m_head) # nn.Sequential是torch中的一个顺序容器
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        #================================================模型结束=============================================#



    # 定义模型的前向传播流程
    def forward(self, x):
        x = self.sub_mean(x) # 先让所有像素值减去平均值，然后进行处理
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x) # 将所有像素值加回来

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

