import argparse
import torch

import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import my_utils as my

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


if __name__ == '__main__':
    # 用于解析参数
    # 这里需要注意，测试的时候，需要输入的是yuv文件，而不是做好的h5数据集
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--hr-tex-yuv-path', type=str, required=True)
    parser.add_argument('--hr-occ-yuv-path', type=str, required=True)
    parser.add_argument('--lr-tex-yuv-path', type=str, required=True)
    parser.add_argument('--lr-occ-yuv-path', type=str, required=True)
    parser.add_argument('--out-file', type=str, required=True) # 输出图像的位置
    # parser.add_argument('--scale', type=int, default=1)
    args = parser.parse_args()

    
    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device) # 用于指定model加载到某个设备
    state_dict = model.state_dict() # state_dict()是一个类似字典的结构，用于储存需要学习的参数和偏移值

    # torch.load(f, map_location=lambda storage, loc: storage)把f加载到GPU 1中，下述过程用于加载模型参数
    # items()以列表返回可遍历的(键, 值) 元组数组
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p) # 将文件中的参数应用于模型
        else:
            raise KeyError(n)

    # model.train()是启用BN和dropout，model.eval是关闭train和dropout即保证BN层的均值和方差不变
    model.eval()
    
    # 注意该函数的使用，受到路径名称的限制，所以需要路径名符合某一格式
    # 对应的数据已经出存在yuv_to_test_network文件夹中
    hr_tex_y = my.get_n_Ychannel(args.hr_tex_yuv_path)
    hr_occ_y = my.get_n_Ychannel(args.hr_occ_yuv_path)
    lr_tex_y = my.get_n_Ychannel(args.lr_tex_yuv_path)
    lr_occ_y = my.get_n_Ychannel(args.lr_occ_yuv_path)

    for n in range(0,len(hr_tex_y)) : 
        hr_tex = hr_tex_y[n] # 提取一帧hr_tex
        lr_tex = lr_tex_y[n] # 提取一帧lr_tex
        hr_occ = hr_occ_y[int(n/2)] # 提取一帧hr_occ
        lr_occ = lr_occ_y[int(n/2)] # 提取一帧lr_occ
        
        # 数据类型转换
        hr_tex = np.array(hr_tex).astype(np.float32)
        lr_tex = np.array(lr_tex).astype(np.float32)
        hr_occ = np.array(hr_occ).astype(np.float32)
        lr_occ = np.array(lr_occ).astype(np.float32)

        # 将occupancy变得和texture一样大小
        hr_occ = my.ndarray_nearest_neighbour_scaling(hr_occ, len(hr_tex), len(hr_tex[0]))
        lr_occ = my.ndarray_nearest_neighbour_scaling(lr_occ, len(lr_tex), len(lr_tex[0]))
        
        # 拼接occupancy和texture，获取对应的图像数据
        lr = np.array([lr_tex,lr_occ])
        hr = np.array([hr_tex,hr_occ])
        y_lr = lr / 255.
        y_hr = hr / 255.

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
        # 储存生成的图像，将args.scale添加到_bicubic_x后面，然后替换{}，获得新名字
        # output.save(args.image_lr.replace('.', '_srcnn_x{}.'.format(args.scale)))
        output.save(args.image_out)
