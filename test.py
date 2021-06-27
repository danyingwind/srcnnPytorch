import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


if __name__ == '__main__':
    #用于解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    # https://www.cnblogs.com/captain-dl/p/11938864.html
    # 如果网络的输入数据维度或类型上变化不大，也就是每次训练的图像尺寸都是一样的时候，设置为true可以找到最适合当前的高效算法
    # 但如果每次迭代时，输入的数据维度都会变化，那么每次输入的时候都要重新查找最佳，反而会降低效率
    cudnn.benchmark = True

    # 可以在cuda和cpu之间进行选择，并指定设备号，设备号不存在的时候使用当前设备。下列语句的意思是，用设备号为0的cuda。
    # cuda是一种硬件加速程序，https://www.cnblogs.com/AngelaSunny/p/7845587.html
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 用于指定model加载到某个设备
    model = SRCNN().to(device)
 
    # state_dict()是一个类似字典的结构，用于储存需要学习的参数和偏移值
    state_dict = model.state_dict()

    # torch.load(f, map_location=lambda storage, loc: storage)把f加载到GPU 1中，下述过程用于加载模型参数
    # items()以列表返回可遍历的(键, 值) 元组数组
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)#将文件中的参数应用于模型
        else:
            raise KeyError(n)

    # model.train()是启用BN和dropout，model.eval是关闭train和dropout即保证BN层的均值和方差不变
    model.eval()
    
    # 读取图像文件并转换为RGB
    image = pil_image.open(args.image_file).convert('RGB')

    # 文件的上下采样
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    
    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
    # 上述代码的意思是将args.scale添加到_bicubic_x后面，然后替换{}

    # 初始化数组并转换数据类型
    image = np.array(image).astype(np.float32)
    #将RGB转换成YCbCr
    ycbcr = convert_rgb_to_ycbcr(image)

    # 在ycbcr中，0维对应的是Y分量
    y = ycbcr[..., 0]
    y /= 255.

    # torch.from_numpy()完成从array到tensor的转换
    y = torch.from_numpy(y).to(device)

    # unsqueeze(0)表示在维度0，也就是最外层新增1维比如[2,3]变为[1,2,3],再来一次就是变成[1,1,2,3]
    y = y.unsqueeze(0).unsqueeze(0)

    # torch.no_grad() 是一个上下文管理器,表示执行计算，但该计算不会在反向传播中被记录。
    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0) #torch.clamp(input, min, max, out=None)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))# 格式化输出的函数

    # .cpu()表示将数据转移到cpu
    # t.numpy()将tensor转换为ndarray，t可以tensor/标量/向量，转换前后dtype不变
    # squeeze 表示去除最外层的维度，比如[1,1,2]可以经过两次squeeze变为[2]
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    # transpose用于进行数据位置的变换，https://blog.csdn.net/weixin_44438120/article/details/106761480
    # 下述意思为新的0、1、2是由过去的1、2、0得到
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])

    # 将超过上下限的值用上下限来替代，使数值限定在0-255的范围内
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    # 从array得到image
    output = pil_image.fromarray(output) 
    # 储存生成的图像
    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))
