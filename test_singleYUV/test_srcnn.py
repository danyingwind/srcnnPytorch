import argparse
from time import *
import torch
import torch.backends.cudnn as cudnn
import numpy as np

import sys
sys.path.append("..")
import my_utils as my
from model.srcnn_model import SRCNN
from utils import  calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--lr-tex-yuv-path', type=str, required=True) # 高分辨率图像的位置
    parser.add_argument('--hr-tex-yuv-path', type=str, required=True) # 低分辨率图像的位置
    parser.add_argument('--outputHR-path', type=str, required=True) # 输出图像的位置
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
    
    # 注意该函数的使用，受到路径名称的限制，所以需要路径名符合某一格式，对应的数据已经储存在yuv_to_test_network文件夹中
    lr_tex_y,lr_tex_u,lr_tex_v = my.get_n_YUVchannel(args.lr_tex_yuv_path)
    hr_tex_y,hr_tex_u,hr_tex_v = my.get_n_YUVchannel(args.hr_tex_yuv_path)
    out_y = []
    psnr_total = 0
    psnr_lr = 0
    im_cnt = len(lr_tex_y) # 记录帧数
    begin_time = time()
    for n in range(0,im_cnt) : 
        # print("正在处理第{}帧".format(n))
        lr_tex = lr_tex_y[n] # 提取一帧lr_tex
        hr_tex = hr_tex_y[n] # 提取一帧hr_tex
        # 数据类型转换
        lr_tex = np.array(lr_tex).astype(np.float32)
        # 拼接occupancy和texture，获取对应的图像数据
        lr = np.array([lr_tex])
        y_lr = lr / 255.
        # torch.from_numpy()完成从array到tensor的转换
        y_lr = torch.from_numpy(y_lr).to(device)
        # print("y_lr.shape = ", y_lr.shape)
        y_lr = y_lr.unsqueeze(0)
        with torch.no_grad(): # torch.no_grad() 是一个上下文管理器,表示执行计算，但该计算不会在反向传播中被记录。
            preds = model(y_lr).clamp(0.0, 1.0) 


        preds = preds.data.cpu().numpy() # 转换成numpy
        preds_tex_y = preds[0][0]
        # if n == 1:
        #     print("preds.shape = ", preds.shape)
        #     print("preds_tex_y.shape = ", preds_tex_y.shape)
        #     print("hr_tex.shape = ", hr_tex.shape)
        #     print("lr_occ.shape = ", lr_occ.shape)

        lr_tex = lr_tex / 255.
        hr_tex = hr_tex / 255.
        psnr_total += calc_psnr(torch.from_numpy(preds_tex_y),torch.from_numpy(hr_tex) )
        psnr_lr += calc_psnr(torch.from_numpy(lr_tex),torch.from_numpy(hr_tex) )
        preds_tex_y = preds_tex_y*255
        preds_tex_y = preds_tex_y.astype(np.uint8)
        out_y.append(preds_tex_y)
    end_time = time()  

    print('PSNR psnr_total: {:.2f}'.format(psnr_total/im_cnt))# 格式化输出的函数
    print('PSNR psnr_lr: {:.2f}'.format(psnr_lr/im_cnt))# 格式化输出的函数
    print('模块运行时间 = ', end_time-begin_time)


    # transpose用于进行数据位置的变换，https://blog.csdn.net/weixin_44438120/article/details/106761480
    # 这里还需要在读入数据的时候保存uv，用来合成yuv
    # output = np.array([preds, ycbcr_lr[..., 1], ycbcr_lr[..., 2]]).transpose([1, 2, 0])

    # 把YUV通道的数据组合起来生成YUV文件
    # print("进行数据类型转化")
    # out_y = out_y.data.cpu().numpy()
    print("开始进行文件写入")
    # print("shape of out_y[0] = ",out_y[0].shape)
    # print("shape of lr_tex_u[0] = ",lr_tex_u[0].shape)
    # print("shape of lr_tex_v[0] = ",lr_tex_v[0].shape)
    # print("len of out_y = ",len(out_y))
    # print("len of lr_tex_u = ",len(lr_tex_u))
    # print("len of lr_tex_v = ",len(lr_tex_v))
    # print("img_cnt = ",im_cnt)
    frame_num,frame_width,frame_height = my.get_yuv_paras(args.lr_tex_yuv_path) # 这里只要frame_num
    # print(frame_num,frame_width,frame_height)
    my.writeyuv(out_y,hr_tex_u,hr_tex_v, frame_num, args.outputHR_path)
    print("完成处理")   