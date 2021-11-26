import argparse
from time import *
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

import sys
sys.path.append("..")
from model.srcnn_model import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-hr', type=str, required=True) # 高分辨率图像的位置
    parser.add_argument('--image-lr', type=str, required=True) # 低分辨率图像的位置
    parser.add_argument('--image-out', type=str, required=True) # 输出图像的位置
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
 
    # state_dict()是一个类似字典的结构，用于储存需要学习的参数和偏移值
    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)#将文件中的参数应用于模型
        else:
            raise KeyError(n)

    model.eval()
    
    # 读取图像文件并转换为RGB
    # image = pil_image.open(args.image_file).convert('RGB')
    lr = pil_image.open(args.image_lr ).convert('RGB')
    hr = pil_image.open(args.image_hr ).convert('RGB')


    # 初始化数组并转换数据类型
    lr = np.array(lr).astype(np.float32)
    hr = np.array(hr).astype(np.float32)
    #将RGB转换成YCbCr
    ycbcr_lr = convert_rgb_to_ycbcr(lr)
    ycbcr_hr = convert_rgb_to_ycbcr(hr)

    # 在ycbcr中，0维对应的是Y分量
    y_lr = ycbcr_lr[..., 0]
    y_lr /= 255.
    y_hr = ycbcr_hr[..., 0]
    y_hr /= 255.

    # torch.from_numpy()完成从array到tensor的转换
    y_lr = torch.from_numpy(y_lr).to(device)
    y_hr = torch.from_numpy(y_hr).to(device)

    # unsqueeze(0)表示在维度0，也就是最外层新增1维比如[2,3]变为[1,2,3],再来一次就是变成[1,1,2,3]
    y_lr = y_lr.unsqueeze(0).unsqueeze(0)
    y_hr = y_hr.unsqueeze(0).unsqueeze(0)

    # torch.no_grad() 是一个上下文管理器,表示执行计算，但该计算不会在反向传播中被记录。
    with torch.no_grad():
        preds = model(y_lr).clamp(0.0, 1.0) 
        # torch.clamp(input, min, max, out=None)
        # 这里在(0,1)进行clamp，是因为图像输入的时候已经使用255来将范围调整到(0,1)

    # psnr = calc_psnr(y_hr, preds)
    # print('PSNR: {:.2f}'.format(psnr))# 格式化输出的函数
    psnr_hr = calc_psnr(y_hr, preds)
    psnr_lr = calc_psnr(y_hr, y_lr)
    print('PSNR hr2preds: {:.2f}'.format(psnr_hr))# 格式化输出的函数
    print('PSNR hr2lr: {:.2f}'.format(psnr_lr))# 格式化输出的函数

    # .cpu()表示将数据转移到cpu
    # t.numpy()将tensor转换为ndarray，t可以tensor/标量/向量，转换前后dtype不变
    # squeeze 表示去除最外层的维度，比如[1,1,2]可以经过两次squeeze变为[2]
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    # transpose用于进行数据位置的变换，https://blog.csdn.net/weixin_44438120/article/details/106761480
    # 下述意思为新的0、1、2是由过去的1、2、0得到
    output = np.array([preds, ycbcr_lr[..., 1], ycbcr_lr[..., 2]]).transpose([1, 2, 0])

    # 将超过上下限的值用上下限来替代，使数值限定在0-255的范围内
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    # 从array得到image
    output = pil_image.fromarray(output) 
    # 储存生成的图像
    # output.save(args.image_lr.replace('.', '_srcnn_x{}.'.format(args.scale)))
    output.save(args.image_out)
