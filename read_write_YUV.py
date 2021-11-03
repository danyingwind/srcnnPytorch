import argparse
import torch

import torch.backends.cudnn as cudnn
import numpy as np
import my_utils as my

from models import SRCNN
from utils import calc_psnr


if __name__ == '__main__':
    # 这里需要注意，测试的时候，需要输入的是yuv文件，而不是做好的h5数据集
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    print("进行YUV读入")
    input_y,input_u,input_v = my.get_n_YUVchannel(args.input)
    print("获取YUV参数")
    frame_num,frame_width,frame_height = my.get_yuv_paras(args.input)
    print("进行YUV写入")
    my.writeyuv(input_y, input_u, input_v, frame_num, args.output)
    print("完成处理")