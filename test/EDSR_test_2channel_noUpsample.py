import argparse
import torch

import torch.backends.cudnn as cudnn
import numpy as np
import my_utils as my

import sys
sys.path.append("..")
from model.srcnnEDSR_model_noUpsample import srcnnEDSR
from utils import calc_psnr


if __name__ == '__main__':
    # 这里需要注意，测试的时候，需要输入的是yuv文件，而不是做好的h5数据集
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--num_colors', type=int, default=2)
    parser.add_argument('--lr-tex-yuv-path', type=str, required=True)
    parser.add_argument('--lr-occ-yuv-path', type=str, required=True)
    parser.add_argument('--hr-tex-yuv-path', type=str, required=True)
    parser.add_argument('--outputLR-path', type=str, required=True)
    parser.add_argument('--outputHR-path', type=str, required=True)
    # 初始化模型需要的参数
    parser.add_argument('--num_resblocks', type=int, default=8) # 需要修改
    parser.add_argument('--num_feats', type=int, default=16) # 需要修改
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = srcnnEDSR(args).to(device) # 用于指定model加载到某个设备
    state_dict = model.state_dict() # state_dict()是一个类似字典的结构，用于储存需要学习的参数和偏移值
    # for key in state_dict.keys():
    #     print(key+", shape = " + str(state_dict[key].shape))
    weights = torch.load(args.weights_file, map_location=lambda storage, loc: storage)
    # for key in weights:
    #     print(key+", shape = " + str(weights[key].shape))

    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p) # 将文件中的参数应用于模型
        else:
            raise KeyError(n)

    # model.train()是启用BN和dropout，model.eval是关闭train和dropout即保证BN层的均值和方差不变
    model.eval()
    
    # 注意该函数的使用，受到路径名称的限制，所以需要路径名符合某一格式，对应的数据已经储存在yuv_to_test_network文件夹中
    # # 低清晰度拼接
    # lr_tex_y,lr_tex_u,lr_tex_v = my.get_n_YUVchannel(args.lr_tex_yuv_path)
    hr_tex_y,hr_tex_u,hr_tex_v = my.get_n_YUVchannel(args.hr_tex_yuv_path)
    lr_occ_y = my.get_n_Ychannel(args.lr_occ_yuv_path)
    out_y = []
    psnr_total = 0
    psnr_valid = 0
    psnr_lr = 0
    im_cnt = len(hr_tex_y) # 记录帧数
    for n in range(0,im_cnt) : 
        # print("正在处理第{}帧".format(n))
        lr_tex = hr_tex_y[n] # 提取一帧lr_tex
        lr_occ = lr_occ_y[int(n/2)] # 提取一帧lr_occ
        hr_tex = hr_tex_y[n] # 提取一帧hr_tex
        # 数据类型转换
        lr_tex = np.array(lr_tex).astype(np.float32)
        lr_occ = np.array(lr_occ).astype(np.float32)
        # 将occupancy变得和texture一样大小
        lr_occ = my.ndarray_nearest_neighbour_scaling(lr_occ, len(lr_tex), len(lr_tex[0]))
        # 拼接occupancy和texture，获取对应的图像数据
        lr = np.array([lr_tex,lr_occ])
        y_lr = lr / 255.
        # torch.from_numpy()完成从array到tensor的转换
        y_lr = torch.from_numpy(y_lr).to(device)
        # print("y_lr.shape = ", y_lr.shape)
        y_lr = y_lr.unsqueeze(0)
        with torch.no_grad(): # torch.no_grad() 是一个上下文管理器,表示执行计算，但该计算不会在反向传播中被记录。
            preds = model(y_lr).clamp(0.0, 1.0) 

        preds = preds.data.cpu().numpy() # 转换成numpy
        preds_tex_y = preds[0][0]
        if n == 1:
            print("preds.shape = ", preds.shape)
            print("preds_tex_y.shape = ", preds_tex_y.shape)
            print("hr_tex.shape = ", hr_tex.shape)
            print("lr_occ.shape = ", lr_occ.shape)

        lr_tex = lr_tex / 255.
        hr_tex = hr_tex / 255.
        psnr_total += calc_psnr(torch.from_numpy(preds_tex_y),torch.from_numpy(hr_tex) )
        psnr_valid += calc_psnr(torch.from_numpy(preds_tex_y*lr_occ),torch.from_numpy(hr_tex*lr_occ) )
        psnr_lr += calc_psnr(torch.from_numpy(lr_tex),torch.from_numpy(hr_tex) )
        preds_tex_y = preds_tex_y*255
        preds_tex_y = preds_tex_y.astype(np.uint8)
        out_y.append(preds_tex_y)
        

    # print('PSNR hr2preds_total（采用lr的occ，但训练用的是hr的occ）: {:.2f}'.format(psnr_total/im_cnt))# 格式化输出的函数
    print('PSNR psnr_total: {:.2f}'.format(psnr_total/im_cnt))# 格式化输出的函数
    print('PSNR psnr_valid: {:.2f}'.format(psnr_valid/im_cnt))# 格式化输出的函数
    print('PSNR psnr_lr: {:.2f}'.format(psnr_lr/im_cnt))# 格式化输出的函数


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
    # # 低清晰度输出
    # my.writeyuv(out_y,lr_tex_u,lr_tex_v, frame_num, args.outputLR_path)
    my.writeyuv(out_y,hr_tex_u,hr_tex_v, frame_num, args.outputHR_path)
    print("完成处理")

