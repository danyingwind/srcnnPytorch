import numpy as np
import argparse
import glob
import math

#---读取1帧的Y分量---
def get_Ychannel(f_stream,frame_width, frame_height, patch_size=2):
    y_buf = f_stream.read(frame_width * frame_height)
    u_buf = f_stream.read(frame_width * frame_height//4)
    v_buf = f_stream.read(frame_width * frame_height//4)
    dataY = np.frombuffer(y_buf, dtype = np.uint8)
    dataY = dataY.reshape(frame_height, frame_width) 
    dataY = dataY.astype(np.uint8)

    return dataY

#---读取n帧的Y分量---
def get_n_Ychannel(yuv_path,patch_size = 2):
    frame_num, frame_width, frame_height = get_yuv_paras(yuv_path)
    f_stream = open(yuv_path, 'rb')
    Y_data = []
    for k in range(frame_num):
        Y = get_Ychannel(f_stream, frame_width, frame_height)
        Y_data.append(Y)
    return Y_data

# 用于通过最近邻居插值法对array进行放大
def ndarray_nearest_neighbour_scaling(label, new_h, new_w):
    if len(label.shape) == 2:
        label_new = np.zeros([new_h, new_w], dtype=label.dtype)
    else:
        label_new = np.zeros([new_h, new_w, label.shape[2]], dtype=label.dtype)

    scale_h = new_h / label.shape[0]
    scale_w = new_w / label.shape[1]

    y_pos = np.arange(new_h)
    x_pos = np.arange(new_w)
    y_pos = np.floor(y_pos / scale_h).astype(np.int32)
    x_pos = np.floor(x_pos / scale_w).astype(np.int32)

    y_pos = y_pos.reshape(y_pos.shape[0], 1)
    y_pos = np.tile(y_pos, (1, new_w))
    x_pos = np.tile(x_pos, (new_h, 1))
    assert y_pos.shape == x_pos.shape

    label_new[:, :] = label[y_pos[:, :], x_pos[:, :]]
    return label_new

# 获取R5_yuv/texture下的所有yuv的路径信息
def get_path_lists(yuv_path):
    # 这里的paths有三部分，对应seq23,seq24,seq25
    # 例子'/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture/seq23'
    paths = sorted(glob.glob('{}/*'.format(yuv_path)))
    path_lists = []
    for path in paths:
        sub_path_list = sorted(glob.glob('{}/*'.format(path)))
        path_lists.extend(sub_path_list)

    # path_lists的每个元素为
    # '/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture/seq23/S23C2AIR05_F300_GOF0_texture_1280x1280_8bit_p420.yuv'
    return path_lists

# 输入为'/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture/seq23/S23C2AIR05_F300_GOF0_texture_1280x1280_8bit_p420.yuv'
# 获取frame_num, frame_width, frame_height的信息
def get_yuv_paras(path):
    path_split = path.split("/")
    yuv_name = path_split[-1]
    yuv_name_split = yuv_name.split("_")
    GOF_num = yuv_name_split[2][3]
    frame_paras = yuv_name_split[4].split("x")
    frame_width = int(frame_paras[0])
    frame_height = int(frame_paras[1])
    frame_num = 0
    if(GOF_num >= '0' and GOF_num <= '8'):
        if(path_split[6] == "texture"):
            frame_num = 64
        elif(path_split[6] == "occupancy"):
            frame_num = 32
    elif(GOF_num == 9):
        if(path_split[6] == "texture"):
            frame_num = 24
        elif(path_split[6] == "occupancy"):
            frame_num = 12
    return [frame_num,frame_width,frame_height]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yuv-path', type=str, required=True)
    # 从默认的地方获取数据，并写入默认的数据结构
    args = parser.parse_args()
    ans = get_path_lists(args.yuv_path)
    paras = get_yuv_paras(ans[0])
    print(len(ans))

