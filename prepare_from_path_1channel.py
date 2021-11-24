import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image # python的一个图像处理模块
import my_utils as my

# 本文件生成的数据集只有texture，不含occupancy
# lr对应的是R1_rec，hr对应的是R5原始
# 本文件生成的数据集只涉及yuv

# 训练集的数据预处理
# 这里传入的是一个列表，列表里储存的是如下格式的路径，对应的是hr_tex_yuv_path
# '/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture/seq23/S23C2AIR05_F300_GOF0_texture_1280x1280_8bit_p420.yuv'
# 需要由该路径生成如下路径

def train(paths):
    h5_file = h5py.File(args.output_path, 'w')# 以写的方式打开路径文件

    lr_patches = []
    hr_patches = []

    # 获取文件的yuv列表
    train_paths_hrtex = paths[0]
    train_paths_lrtex = paths[1]

    # 开始依次处理所有的yuv
    for m in range(len(train_paths_hrtex)):
        # if(m > 0) : #这里只拿一个yuv做测试
        #     break
        # 获取第一组yuv，每一组应当包含2个yuv，对应的是同一个点云GOF，高清纹理/低清纹理
        hr_tex_yuv_path = train_paths_hrtex[m]
        lr_tex_yuv_path = train_paths_lrtex[m]
    
        # 根据路径对应的文件获取y分量
        hr_tex_y = my.get_n_Ychannel(hr_tex_yuv_path)
        lr_tex_y = my.get_n_Ychannel(lr_tex_yuv_path)

        # 每个完整的GOF中有32帧点云，对应64帧纹理和32帧占用，处理GOF中的每一帧
        for n in range(0,len(hr_tex_y)) : 
            hr_tex = hr_tex_y[n] # 提取一帧hr_tex
            lr_tex = lr_tex_y[n] # 提取一帧lr_tex
            
            # 数据类型转换
            hr_tex = np.array(hr_tex).astype(np.float32)
            lr_tex = np.array(lr_tex).astype(np.float32)
        
            # 获取每张图的lr_tex_patches,hr_tex_patches
        #################注意这里有中括号，原本是拼接occupancy和texture####################
            for i in range(0, hr_tex.shape[0] - args.patch_size + 1, args.stride): 
                for j in range(0, hr_tex.shape[1] - args.patch_size + 1, args.stride):
                    lr_patches.append(np.array([lr_tex[i:i + args.patch_size, j:j + args.patch_size]]))
                    hr_patches.append(np.array([hr_tex[i:i + args.patch_size, j:j + args.patch_size]]))

    # 经过上述for循环，已经获得了所有图像分割后的patch，将patches转换成array
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)


    # 创建数据集，注意这里的dataset
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    # 关闭文件
    h5_file.close()

# 验证集的数据预处理
def eval(paths):
    h5_file = h5py.File(args.output_path, 'w') # 以写的方式按路径打开文件

    # 创建文件夹，注意这里是group
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    train_paths_hrtex = paths[0]
    train_paths_lrtex = paths[1]

    idx = 0 # 这里的idx是数据集的索引编号
    for m in range(len(train_paths_hrtex)):
        # if(m > 0) : #这里只拿一个yuv做测试
        #     break
        hr_tex_yuv_path = train_paths_hrtex[m]
        lr_tex_yuv_path = train_paths_lrtex[m]
    
        # 根据路径对应的文件获取y分量
        hr_tex_y = my.get_n_Ychannel(hr_tex_yuv_path)
        lr_tex_y = my.get_n_Ychannel(lr_tex_yuv_path)

        for n in range(0,len(hr_tex_y)) : 
            hr_tex = hr_tex_y[n] # 提取一帧hr_tex
            lr_tex = lr_tex_y[n] # 提取一帧lr_tex
            
            # 数据类型转换
            hr_tex = np.array(hr_tex).astype(np.float32)
            lr_tex = np.array(lr_tex).astype(np.float32)
            
            
            #################注意这里有中括号，原本是拼接occupancy和texture####################
            lr = np.array([lr_tex])
            hr = np.array([hr_tex])

            lr_group.create_dataset(str(idx), data=lr)
            hr_group.create_dataset(str(idx), data=hr)
            idx = idx + 1

    h5_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 这里读入的路径是类似于'/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture'
    # 可以由此生成所需的~/R5_yuv/occupancy, ~/R1_yuv_rec/texture, ~/R1_yuv_rec/occupancy
    parser.add_argument('--hrtex-yuv-path', type=str, required=True)
    parser.add_argument('--lrtex-yuv-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True) # 这里指示生成的h5文件的地址
    parser.add_argument('--dataset-type', type=str, required=True) # 这里指示生成训练集or验证集
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=100)    
    args = parser.parse_args()

    train_paths_hrtex, eval_paths_hrtex,test_paths_hrtex = my.get_path_lists(args.hrtex_yuv_path)
    train_paths_lrtex, eval_paths_lrtex,test_paths_lrtex = my.get_path_lists(args.lrtex_yuv_path)
    

    if args.dataset_type == "train":
        # train([train_paths_hrtex,train_paths_lrtex])
        train([test_paths_hrtex,test_paths_lrtex])
    elif args.dataset_type == "eval":
        # eval([eval_paths_hrtex,eval_paths_lrtex])
        eval([test_paths_hrtex,test_paths_lrtex])
    elif args.dataset_type == "test":
        eval([test_paths_hrtex,test_paths_lrtex])
