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

#---读取1帧的YUV分量---
def get_YUVchannel(f_stream,frame_width, frame_height, patch_size=2):
    y_buf = f_stream.read(frame_width * frame_height)
    u_buf = f_stream.read(frame_width * frame_height//4)
    v_buf = f_stream.read(frame_width * frame_height//4)

    dataY = np.frombuffer(y_buf, dtype = np.uint8)
    dataY = dataY.reshape(frame_height, frame_width) 
    dataY = dataY.astype(np.uint8)

    dataU = np.frombuffer(u_buf, dtype = np.uint8)
    dataU = dataU.reshape((int)(frame_height/2),(int)(frame_width/2)) 
    dataU = dataU.astype(np.uint8)

    dataV = np.frombuffer(v_buf, dtype = np.uint8)
    dataV = dataV.reshape((int)(frame_height/2), (int)(frame_width/2)) 
    dataV = dataV.astype(np.uint8)

    return dataY,dataU,dataV

#---读取n帧的Y分量---
def get_n_Ychannel(yuv_path,patch_size = 2):
    frame_num, frame_width, frame_height = get_yuv_paras(yuv_path)
    f_stream = open(yuv_path, 'rb')
    Y_data = []
    for k in range(frame_num):
        Y = get_Ychannel(f_stream, frame_width, frame_height)
        Y_data.append(Y)
    return Y_data

#---读取n帧的YUV分量---
def get_n_YUVchannel(yuv_path,patch_size = 2):
    frame_num, frame_width, frame_height = get_yuv_paras(yuv_path)
    f_stream = open(yuv_path, 'rb')
    Y_data = []
    U_data = []
    V_data = []
    for k in range(frame_num):
        Y,U,V = get_YUVchannel(f_stream, frame_width, frame_height)
        Y_data.append(Y)
        U_data.append(U)
        V_data.append(V)
    return Y_data,U_data,V_data

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

# 获取'/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture'下的所有yuv的路径信息
def get_path_lists(yuv_path):
    # 这里的paths有三部分，对应seq23,seq24,seq25
    # 例子'/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture/seq23'
    paths = sorted(glob.glob('{}/*'.format(yuv_path)))
    path_lists = []
    for path in paths:
        sub_path_list = sorted(glob.glob('{}/*'.format(path)))
        path_lists.extend(sub_path_list)
    
    # 经过上述for循环，可以获取所有的yuv路径
    # 下面将所有路径拆分为训练集0-6，验证集7-8，测试集9
    train_paths = []
    eval_paths = []
    test_paths = []
    for path in path_lists:
        path_split = path.split("/")
        yuv_name = path_split[-1]
        yuv_name_split = yuv_name.split("_")
        GOF_num = yuv_name_split[2][3]
        if(GOF_num >= '4' and GOF_num <= '6'):
            train_paths.append(path)
        elif(GOF_num >= '7' and GOF_num <= '8'):
            eval_paths.append(path)
        elif(GOF_num == '9'):
            test_paths.append(path)

    return [train_paths,eval_paths,test_paths]

    # path_lists的每个元素为
    # '/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture/seq23/S23C2AIR05_F300_GOF0_texture_1280x1280_8bit_p420.yuv'
    # return path_lists

# 输入为'/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture/seq23/S23C2AIR05_F300_GOF0_texture_1280x1280_8bit_p420.yuv'
# 获取frame_num, frame_width, frame_height的信息
def get_yuv_paras(path):
    path_split = path.split("/")
    yuv_name = path_split[-1]
    yuv_name_split = yuv_name.split("_")
    GOF_num = yuv_name_split[2][3]
    frame_paras = [0,0]
    if(yuv_name_split[4] == "rec"):
        frame_paras = yuv_name_split[5].split("x")
    else :
        frame_paras = yuv_name_split[4].split("x")
    frame_width = int(frame_paras[0])
    frame_height = int(frame_paras[1])
    frame_num = 0
    if(GOF_num >= '0' and GOF_num <= '8'):
        if(path_split[6] == "texture"):
            frame_num = 64
        elif(path_split[6] == "occupancy"):
            frame_num = 32
    elif(GOF_num == '9'):
        if(path_split[6] == "texture"):
            frame_num = 24
        elif(path_split[6] == "occupancy"):
            frame_num = 12
    elif(GOF_num == 'p'): # 当采用单帧测试时，命名为GOFp
        if(path_split[6] == "texture"):
            frame_num = 2
        elif(path_split[6] == "occupancy"):
            frame_num = 1
    return [frame_num,frame_width,frame_height]

# def writeyuv(Ylist, Ulist, Vlist, TotalFrames, FilenameOut, FrameSizeY):
def writeyuv(Ylist, Ulist, Vlist, TotalFrames, FilenameOut):
    with open(FilenameOut, 'wb+') as f:
        for frameidx in range(TotalFrames):
            # FrameYbuf = FrameSizeY
            # FrameUVbuf = FrameSizeY / 4
            # Ybufstart = FrameYbuf * frameidx
            # Ybufend = FrameYbuf * (frameidx + 1)
            #for Yindex in range(Ybufstart, Ybufend):
            f.write(Ylist[frameidx])
            # UVbufstart = FrameUVbuf * frameidx
            # UVbufend = FrameUVbuf * (frameidx + 1)
            # for Uindex in range(UVbufstart, UVbufend):
            f.write(Ulist[frameidx])
            # for Vindex in range(UVbufstart, UVbufend):
            f.write(Vlist[frameidx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yuv-path', type=str, required=True)
    # 从默认的地方获取数据，并写入默认的数据结构
    args = parser.parse_args()
    train_paths, eval_paths,test_paths = get_path_lists(args.yuv_path)

    print(len(train_paths),len(eval_paths),len(test_paths),sep=',')









