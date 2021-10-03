import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image# python的一个图像处理模块
import my_utils as my

# 本文件生成的数据集含texture+occupancy
# lr对应的是R1_rec，hr对应的是R5原始
# 本文件生成的数据集只涉及yuv

# 训练集的数据预处理
# 这里传入的是一个列表，列表里储存的是如下格式的路径，对应的是hr_tex_yuv_path
# '/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture/seq23/S23C2AIR05_F300_GOF0_texture_1280x1280_8bit_p420.yuv'
# 需要由该路径生成如下路径

def train(paths):
    h5_file = h5py.File(args.output_path, 'w')# 以写的方式打开路径文件
    # https://blog.csdn.net/qq_34859482/article/details/80115237
    # h5py是1个管理数据的容器，里面可以有具体的数据以及数据分组

    lr_patches = []
    hr_patches = []

    train_paths_hrtex = paths[0]
    train_paths_hrocc = paths[1]
    train_paths_lrtex = paths[2]
    train_paths_lrocc = paths[3]


    for m in range(len(train_paths_hrtex)):
        if(m > 0) : #这里只拿一个yuv做测试
            break
        hr_tex_yuv_path = train_paths_hrtex[m]
        hr_occ_yuv_path = train_paths_hrocc[m]
        lr_tex_yuv_path = train_paths_lrtex[m]
        lr_occ_yuv_path = train_paths_lrocc[m]
    
        # 根据路径对应的文件获取y分量
        hr_tex_y = my.get_n_Ychannel(hr_tex_yuv_path)
        hr_occ_y = my.get_n_Ychannel(hr_occ_yuv_path)
        lr_tex_y = my.get_n_Ychannel(lr_tex_yuv_path)
        lr_occ_y = my.get_n_Ychannel(lr_occ_yuv_path)

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

            # 通过观察的中间文件可以发现，占用图的大小始终小于texture，
            # 因此在处理的时候需要先将occupancy变得和texture一样大小
            hr_occ2 = my.ndarray_nearest_neighbour_scaling(hr_occ, len(hr_tex[0]), len(hr_tex[1]))
            lr_occ = my.ndarray_nearest_neighbour_scaling(lr_occ, len(lr_tex[0]), len(lr_tex[1]))
            
        
            # CNN处理图像的时候，不是一次输入一整块，而是把图像划分成多个patch
            # 获取每张图的lr_tex_patches,hr_tex_patches,lr_occ_patches,hr_occ_patches
            for i in range(0, hr_tex.shape[0] - args.patch_size + 1, args.stride): 
                for j in range(0, hr_tex.shape[1] - args.patch_size + 1, args.stride):
                    lr_patches.append(np.array([lr_tex[i:i + args.patch_size, j:j + args.patch_size],lr_occ[i:i + args.patch_size, j:j + args.patch_size]]))
                    hr_patches.append(np.array([hr_tex[i:i + args.patch_size, j:j + args.patch_size],hr_occ[i:i + args.patch_size, j:j + args.patch_size]]))

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
    train_paths_hrocc = paths[1]
    train_paths_lrtex = paths[2]
    train_paths_lrocc = paths[3]

    idx = 0
    for m in range(len(train_paths_hrtex)):
        if(m > 0) : #这里只拿一个yuv做测试
            break
        hr_tex_yuv_path = train_paths_hrtex[m]
        hr_occ_yuv_path = train_paths_hrocc[m]
        lr_tex_yuv_path = train_paths_lrtex[m]
        lr_occ_yuv_path = train_paths_lrocc[m]
    
        # 根据路径对应的文件获取y分量
        hr_tex_y = my.get_n_Ychannel(hr_tex_yuv_path)
        hr_occ_y = my.get_n_Ychannel(hr_occ_yuv_path)
        lr_tex_y = my.get_n_Ychannel(lr_tex_yuv_path)
        lr_occ_y = my.get_n_Ychannel(lr_occ_yuv_path)

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

            # 通过观察的中间文件可以发现，占用图的大小始终小于texture，
            # 因此在处理的时候需要先将occupancy变得和texture一样大小
            hr_occ = my.ndarray_nearest_neighbour_scaling(hr_occ, len(hr_tex[0]), len(hr_tex[1]))
            lr_occ = my.ndarray_nearest_neighbour_scaling(lr_occ, len(lr_tex[0]), len(lr_tex[1]))
            
            # 拼接occupancy和texture
            lr = np.array([lr_tex,lr_occ])
            hr = np.array([hr_tex,hr_occ])

            # 这里得到hr和lr的方式和大小都是和train一样，但是生成数据集的方式不一样
            # train是生成patch，然后用patch创建数据集，一个hr数据集，一个lr数据集
            # eval是生成group，然后对于每个image_dir创建一个数据集，数据集中同时具有hr和lr
            lr_group.create_dataset(str(idx), data=lr)
            hr_group.create_dataset(str(idx), data=hr)
            idx = idx + 1

    h5_file.close()

# 每个python文件都可能直接作为脚本执行or作为一个包导入到其他文件中
# 只有当该文件直接作为脚本执行的时候，if __name__ == '__main__'才判断为true
if __name__ == '__main__':
    # https://blog.csdn.net/qq_36293056/article/details/105122682
    # 创建一个解析器，其中有将命令行解析成python数据所需要的所有信息
    parser = argparse.ArgumentParser()
    # 这里读入的路径是类似于'/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture'
    # 可以由此生成所需的~/R5_yuv/occupancy, ~/R1_yuv_rec/texture, ~/R1_yuv_rec/occupancy
    parser.add_argument('--hrtex-yuv-path', type=str, required=True)
    parser.add_argument('--hrocc-yuv-path', type=str, required=True)
    parser.add_argument('--lrtex-yuv-path', type=str, required=True)
    parser.add_argument('--lrocc-yuv-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True) # 这里指示生成的h5文件的地址
    parser.add_argument('--dataset-type', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=100)
    # parser.add_argument('--scale', type=int, default=2)
    
    # parser.add_argument('--eval', action='store_true') # 当'--eval'出现时，将其状态设为true
    
    # 从默认的地方获取数据，并写入默认的数据结构
    args = parser.parse_args()

    train_paths_hrtex, eval_paths_hrtex,test_paths_hrtex = my.get_path_lists(args.hrtex_yuv_path)
    train_paths_hrocc, eval_paths_hrocc,test_paths_hrocc = my.get_path_lists(args.hrocc_yuv_path)
    train_paths_lrtex, eval_paths_lrtex,test_paths_lrtex = my.get_path_lists(args.lrtex_yuv_path)
    train_paths_lrocc, eval_paths_lrocc,test_paths_lrocc = my.get_path_lists(args.lrocc_yuv_path)

    if args.dataset_type == "train":
        train([train_paths_hrtex,train_paths_hrocc,train_paths_lrtex,train_paths_lrocc])
    elif args.dataset_type == "eval":
        eval([eval_paths_hrtex,eval_paths_hrocc,eval_paths_lrtex,eval_paths_lrocc])
    elif args.dataset_type == "test":
        eval([test_paths_hrtex,test_paths_hrocc,test_paths_lrtex,test_paths_lrocc])
