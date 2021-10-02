import numpy as np
import math


def get_Ychannel(f_stream,frame_width, frame_height, patch_size=2):
    y_buf = f_stream.read(frame_width * frame_height)
    u_buf = f_stream.read(frame_width * frame_height//4)
    v_buf = f_stream.read(frame_width * frame_height//4)
    dataY = np.frombuffer(y_buf, dtype = np.uint8)
    dataY = dataY.reshape(frame_height, frame_width) 
    dataY = dataY.astype(np.uint8)

    return dataY

#---读取n帧---
def get_n_Ychannel(yuv_path, patch_size = 2):
    frame_num, frame_width, frame_height = get_info(yuv_path)
    f_stream = open(yuv_path, 'rb')
    Y_data = []
    for k in range(frame_num):
        Y = get_Ychannel(f_stream, frame_width, frame_height)
        Y_data.append(Y)
    return Y_data

def get_info(yuv_path):
    yuv_path_split = yuv_path.split("/")
    

    return []


