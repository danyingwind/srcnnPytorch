import numpy as np
import math
def get_YUV_for_one_frame(f, frame_width, frame_height, patch_size):
    y_buf = f.read(frame_width * frame_height)
    u_buf = f.read(frame_width * frame_height//4)
    v_buf = f.read(frame_width * frame_height//4)
    frame_heightY = frame_height
    frame_widthY = frame_width
    frame_heightU = int(frame_height/2)
    frame_widthU = int(frame_width/2)
    frame_heightV = int(frame_height/2)
    frame_widthV = int(frame_width/2)
    patch_size_UV = int(patch_size/2)
    
    dataY = np.frombuffer(y_buf, dtype = np.uint8)
    dataY = dataY.reshape(frame_height, frame_width)    
    valid_heightY = math.ceil(frame_height / patch_size) * patch_size
    valid_widthY = math.ceil(frame_width / patch_size) * patch_size
    if valid_heightY > frame_heightY:
        dataY = np.concatenate((dataY, np.zeros((valid_heightY - frame_heightY, frame_widthY))), axis = 0)
    if valid_widthY > frame_widthY:
        dataY = np.concatenate((dataY, np.zeros((valid_heightY, valid_widthY - frame_widthY))), axis = 1)

    dataU = np.frombuffer(u_buf, dtype = np.uint8)
    dataU = dataU.reshape(frame_heightU, frame_widthU)    
    valid_heightU = math.ceil(frame_heightU / patch_size_UV) * patch_size_UV
    valid_widthU = math.ceil(frame_widthU / patch_size_UV) * patch_size_UV
    if valid_heightU > frame_heightU:
        dataU = np.concatenate((dataU, np.zeros((valid_heightU - frame_heightU, frame_widthU))), axis = 0)
    if valid_widthU > frame_widthU:
        dataU = np.concatenate((dataU, np.zeros((valid_heightU, valid_widthU - frame_widthU))), axis = 1)

    dataV = np.frombuffer(v_buf, dtype = np.uint8)
    dataV = dataV.reshape(frame_heightV, frame_widthV)
    valid_heightV = math.ceil(frame_heightV / patch_size_UV) * patch_size_UV
    valid_widthV = math.ceil(frame_widthV / patch_size_UV) * patch_size_UV
    if valid_heightV > frame_heightV:
        dataV = np.concatenate((dataV, np.zeros((valid_heightV - frame_heightV, frame_widthV))), axis = 0)
    if valid_widthV > frame_widthV:
        dataV = np.concatenate((dataV, np.zeros((valid_heightV, valid_widthV - frame_widthV))), axis = 1)

    dataY = dataY.astype(np.uint8)
    dataU = dataU.astype(np.uint8)
    dataV = dataV.astype(np.uint8)

    return dataY, dataU, dataV

#---从文件f中跳过前n_frames_start帧，读取n帧---
def getnframes(f_stream, frame_num, frame_width, frame_height,n_frames_start = 0, patch_size = 2):
    Ytmp = []
    # # 跳过n_frames_start帧
    for k in range(n_frames_start):
        luma, chromaU, chromaV = get_YUV_for_one_frame(f_stream, frame_width, frame_height, patch_size)
    for k in range(frame_num):
        valid_luma, valid_chromaU, valid_chromaV = get_YUV_for_one_frame(f_stream, frame_width, frame_height, patch_size)
        Ytmp.append(valid_luma)
    return Ytmp

def main():

    dataset = []
    yuv_names = []
    for yuv_name in yuv_names:
        f_stream = open(yuv_name, 'rb')
        # 从yuv_name中获取宽高
        frame_width, frame_height= 
        datasettmp = getnframes(f_stream, 10, frame_width, frame_height) #从0帧开始，读10帧
        dataset[idx] = datasettmp