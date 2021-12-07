import argparse
import torch
from time import *
import torch.backends.cudnn as cudnn
import numpy as np

import sys
sys.path.append("..")
import my_utils as my
from model.srcnnEDSR_model_noUpsample import srcnnEDSR
from utils import calc_psnr

def process(args,lr_tex_yuv_path,hr_tex_yuv_path,outputHR_path,lr_occ_yuv_paths):
    cnt = len(lr_tex_yuv_path)
    for i in range(cnt):
        args.lr_tex_yuv_path = lr_tex_yuv_path[i]
        args.hr_tex_yuv_path = hr_tex_yuv_path[i]
        args.outputHR_path = outputHR_path[i]
        lr_occ_yuv_path = lr_occ_yuv_paths[i]
        cudnn.benchmark = True
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        model = srcnnEDSR(args).to(device) # 用于指定model加载到某个设备
        state_dict = model.state_dict() # state_dict()是一个类似字典的结构，用于储存需要学习的参数
        # weights = torch.load(args.weights_file, map_location=lambda storage, loc: storage)
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p) # 将文件中的参数应用于模型
            else:
                raise KeyError(n)
        model.eval()
        
        # 注意该函数的使用，受到路径名称的限制，所以需要路径名符合某一格式，对应的数据已经储存在yuv_to_test_network文件夹中
        lr_tex_y,lr_tex_u,lr_tex_v = my.get_n_YUVchannel(args.lr_tex_yuv_path)
        hr_tex_y,hr_tex_u,hr_tex_v = my.get_n_YUVchannel(args.hr_tex_yuv_path)
        lr_occ_y = my.get_n_Ychannel(lr_occ_yuv_path)
        out_y = []
        psnr_total = 0
        psnr_lr = 0
        psnr_valid = 0
        psnr_valid_lr = 0
        im_cnt = len(lr_tex_y) # 记录帧数
        begin_time = time()
        for n in range(0,im_cnt) : 
            lr_tex = lr_tex_y[n] # 提取一帧lr_tex
            hr_tex = hr_tex_y[n] # 提取一帧hr_tex
            # 数据类型转换
            lr_tex = np.array(lr_tex).astype(np.float32)
            lr = np.array([lr_tex])
            y_lr = lr / 255.
            # torch.from_numpy()完成从array到tensor的转换
            y_lr = torch.from_numpy(y_lr).to(device)
            y_lr = y_lr.unsqueeze(0)
            with torch.no_grad(): # torch.no_grad() 是一个上下文管理器,表示执行计算，但该计算不会在反向传播中被记录。
                preds = model(y_lr).clamp(0.0, 1.0) 

            lr_occ = lr_occ_y[int(n/2)] # 提取一帧lr_occ  
            lr_occ = np.array(lr_occ).astype(np.float32) 
            lr_occ = my.ndarray_nearest_neighbour_scaling(lr_occ, len(lr_tex), len(lr_tex[0]))
            
            preds = preds.data.cpu().numpy() # 转换成numpy
            lr_occ_1  = lr_occ # 对应占用图的有效部分
            lr_occ_0 = lr_occ_1 * (-1) + 1 # 对应占用图的无效部分
            # # 直接把处理过的整幅图用于点云重建
            # preds_tex_y = preds[0][0]
            # 只把处理后的无效部分进行替换，有效部分使用原来的
            y0 = preds[0][0]
            y1 = y_lr.data.cpu().numpy()
            preds_tex_y = y0*lr_occ_0+ y1*lr_occ_1
            lr_tex = lr_tex / 255.
            hr_tex = hr_tex / 255.
            psnr_total += calc_psnr(torch.from_numpy(preds_tex_y),torch.from_numpy(hr_tex) )
            psnr_lr += calc_psnr(torch.from_numpy(lr_tex),torch.from_numpy(hr_tex) )
            psnr_valid += calc_psnr(torch.from_numpy(preds_tex_y*lr_occ),torch.from_numpy(hr_tex*lr_occ) )
            psnr_valid_lr += calc_psnr(torch.from_numpy(lr_tex*lr_occ),torch.from_numpy(hr_tex*lr_occ) )
            preds_tex_y = preds_tex_y*255
            preds_tex_y = preds_tex_y.astype(np.uint8)
            out_y.append(preds_tex_y)
        end_time = time()  

        print('处理的文件：',lr_tex_yuv_path[i].split('/')[-1].split('_')[0])
        print('PSNR psnr_total: {:.2f}'.format(psnr_total/im_cnt))# 格式化输出的函数
        print('PSNR psnr_total_lr: {:.2f}'.format(psnr_lr/im_cnt))# 格式化输出的函数
        print('PSNR psnr_valid: {:.2f}'.format(psnr_valid/im_cnt))# 格式化输出的函数
        print('PSNR psnr_valid_lr: {:.2f}'.format(psnr_valid_lr/im_cnt))# 格式化输出的函数        
        print('模块运行时间 = ', end_time-begin_time)
        frame_num,frame_width,frame_height = my.get_yuv_paras(args.lr_tex_yuv_path) 
        my.writeyuv(out_y,hr_tex_u,hr_tex_v, frame_num, args.outputHR_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default="/home/wangdanying/SRCNN/srcnnPytorch/debug/trainLog/EDSR_1channel_noUpsample/best.pth")
    parser.add_argument('--lr-tex-yuv-path', type=str, default="")
    parser.add_argument('--hr-tex-yuv-path', type=str, default="")
    parser.add_argument('--outputHR-path', type=str, default="")
    # 初始化模型需要的参数
    parser.add_argument('--num_colors', type=int, default=1)
    parser.add_argument('--num_resblocks', type=int, default=8) # 需要修改
    parser.add_argument('--num_feats', type=int, default=16) # 需要修改
    args = parser.parse_args()

    lr_tex_yuv_path=["/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq23/r1/S23C2AIR01_F32_GOF0_texture_rec_1280x1280_8bit_p420.yuv","/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq24/r1/S24C2AIR01_F32_GOF0_texture_rec_1280x1344_8bit_p420.yuv","/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq25/r1/S25C2AIR01_F32_GOF0_texture_rec_1280x1280_8bit_p420.yuv","/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq26/r1/S26C2AIR01_F32_GOF0_texture_rec_1280x1296_8bit_p420.yuv"]
    hr_tex_yuv_path=["/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq23/r1/S23C2AIR01_F32_GOF0_texture_1280x1280_8bit_p420.yuv","/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq24/r1/S24C2AIR01_F32_GOF0_texture_1280x1344_8bit_p420.yuv","/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq25/r1/S25C2AIR01_F32_GOF0_texture_1280x1280_8bit_p420.yuv","/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq26/r5/S26C2AIR05_F32_GOF0_texture_1280x1296_8bit_p420.yuv"]
    lr_occ_yuv_paths = ["/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq23/r1/S23C2AIR01_F32_GOF0_occupancy_rec_320x320_8bit_p420.yuv","/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq24/r1/S24C2AIR01_F32_GOF0_occupancy_rec_320x336_8bit_p420.yuv","/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq25/r1/S25C2AIR01_F32_GOF0_occupancy_rec_320x320_8bit_p420.yuv","/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/testbefore1130/seq26/r1/S26C2AIR01_F32_GOF0_occupancy_rec_320x324_8bit_p420.yuv"]
    outputHR_path=['/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test_allnet/EDSR_test_1channel_noUpsample/seq23/r1_process/S23C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv','/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test_allnet/EDSR_test_1channel_noUpsample/seq24/r1_process/S24C2AIR01_F32_dec_GOF0_texture_rec_1280x1344_8bit_p420.yuv','/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test_allnet/EDSR_test_1channel_noUpsample/seq25/r1_process/S25C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv','/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test_allnet/EDSR_test_1channel_noUpsample/seq26/r1_process/S26C2AIR01_F32_dec_GOF0_texture_rec_1280x1296_8bit_p420.yuv']

    process(args,lr_tex_yuv_path,hr_tex_yuv_path,outputHR_path,lr_occ_yuv_paths)
    print("处理完毕！")