import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image# python的一个图像处理模块
from utils import convert_rgb_to_y

# 本文件生成的数据集只含有texture；lr对应的是R1_rec，hr对应的是R1原始

# 训练集的数据预处理
def train(args):
    h5_file = h5py.File(args.output_path, 'w')# 以写的方式打开路径文件
    # https://blog.csdn.net/qq_34859482/article/details/80115237
    # h5py是1个管理数据的容器，里面可以有具体的数据以及数据分组

    lr_patches = []
    hr_patches = []

    
    # https://blog.csdn.net/GeorgeAI/article/details/81035422
    # glob模块用于查找符合特定规则的文件路径名。
    # 查找文件只用到三个匹配符："*", "?", "[]"。
    # "*"匹配0个或多个字符；"?"匹配单个字符；"[]"匹配指定范围内的字符，如：[0-9]匹配数字。
    # sorted(glob.glob('{}/*'.format(args.images_hr)))的意思是
    # 对以args.images_dir为前缀的所有路径进行排序，for是对每张图进行遍历
    hr_tex_paths = sorted(glob.glob('{}/*'.format(args.hr_tex_images)));
    lr_tex_paths = sorted(glob.glob('{}/*'.format(args.lr_tex_images)));
    hr_occ_paths = sorted(glob.glob('{}/*'.format(args.hr_occ_images)));
    lr_occ_paths = sorted(glob.glob('{}/*'.format(args.lr_occ_images)));
    for i in range(0,len(hr_tex_paths)) : 
        hr_tex = pil_image.open(hr_tex_paths[i]).convert('RGB') # 读取hr_tex文件
        lr_tex = pil_image.open(lr_tex_paths[i]).convert('RGB') # 读取lr_tex文件
        hr_occ = pil_image.open(hr_occ_paths[int(i/2)]).convert('RGB') # 读取hr_occ文件
        lr_occ = pil_image.open(lr_occ_paths[int(i/2)]).convert('RGB') # 读取lr_occ文件
        #如果不用convert('RGB')，会导致图像是4通道RGBA，A是透明度通道，现在用不到

        # 通过观察的中间文件可以发现，占用图的大小始终小于texture，
        # 因此在处理的时候需要先将occupancy变得和texture一样大小
        hr_occ = hr_occ.resize((hr_tex.width, hr_tex.height), resample=pil_image.BICUBIC)
        lr_occ = lr_occ.resize((lr_tex.width, lr_tex.height), resample=pil_image.BICUBIC)

        # 数据类型转换
        hr_tex = np.array(hr_tex).astype(np.float32)
        lr_tex = np.array(lr_tex).astype(np.float32)
        hr_occ = np.array(hr_occ).astype(np.float32)
        lr_occ = np.array(lr_occ).astype(np.float32)
        # 根据RGB求Y
        hr_tex = convert_rgb_to_y(hr_tex)
        lr_tex = convert_rgb_to_y(lr_tex)
        hr_occ = convert_rgb_to_y(hr_occ)
        lr_occ = convert_rgb_to_y(lr_occ)
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
def eval(args):
    h5_file = h5py.File(args.output_path, 'w') # 以写的方式按路径打开文件

    # 创建文件夹，注意这里是group
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    # 获取四类图像的路径
    hr_tex_paths = sorted(glob.glob('{}/*'.format(args.hr_tex_images)));
    lr_tex_paths = sorted(glob.glob('{}/*'.format(args.lr_tex_images)));
    hr_occ_paths = sorted(glob.glob('{}/*'.format(args.hr_occ_images)));
    lr_occ_paths = sorted(glob.glob('{}/*'.format(args.lr_occ_images)));

    # enumerate是枚举函数，i、hr_path分别对应索引下标和路径
    for i in range(0,len(hr_tex_paths)) : 
        hr_tex = pil_image.open(hr_tex_paths[i]).convert('RGB') # 读取hr_tex文件
        lr_tex = pil_image.open(lr_tex_paths[i]).convert('RGB') # 读取lr_tex文件
        hr_occ = pil_image.open(hr_occ_paths[int(i/2)]).convert('RGB') # 读取hr_occ文件
        lr_occ = pil_image.open(lr_occ_paths[int(i/2)]).convert('RGB') # 读取lr_occ文件

        # 通过观察的中间文件可以发现，占用图的大小始终小于texture，
        # 因此在处理的时候需要先将occupancy变得和texture一样大小
        hr_occ = hr_occ.resize((hr_tex.width, hr_tex.height), resample=pil_image.BICUBIC)
        lr_occ = lr_occ.resize((lr_tex.width, lr_tex.height), resample=pil_image.BICUBIC)
       
        # 数据类型转换
        hr_tex = np.array(hr_tex).astype(np.float32)
        lr_tex = np.array(lr_tex).astype(np.float32)
        hr_occ = np.array(hr_occ).astype(np.float32)
        lr_occ = np.array(lr_occ).astype(np.float32)
        # 根据RGB求Y
        hr_tex = convert_rgb_to_y(hr_tex)
        lr_tex = convert_rgb_to_y(lr_tex)
        hr_occ = convert_rgb_to_y(hr_occ)
        lr_occ = convert_rgb_to_y(lr_occ)

        # 拼接occupancy和texture
        lr = np.array([lr_tex,lr_occ])
        hr = np.array([hr_tex,hr_occ])

        # 这里得到hr和lr的方式和大小都是和train一样，但是生成数据集的方式不一样
        # train是生成patch，然后用patch创建数据集，一个hr数据集，一个lr数据集
        # eval是生成group，然后对于每个image_dir创建一个数据集，数据集中同时具有hr和lr
        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()

# 每个python文件都可能直接作为脚本执行or作为一个包导入到其他文件中
# 只有当该文件直接作为脚本执行的时候，if __name__ == '__main__'才判断为true
if __name__ == '__main__':
    # https://blog.csdn.net/qq_36293056/article/details/105122682
    # 创建一个解析器，其中有将命令行解析成python数据所需要的所有信息
    parser = argparse.ArgumentParser()
    #parser.add_argument('--images-dir', type=str, required=True) # 这里train和eval都用了images_dir，不是同一个
    #parser.add_argument('--images-lr', type=str, required=True) # 这里指示低分辨率图像的地址
    #parser.add_argument('--images-hr', type=str, required=True) # 这里指示高分辨率图像的地址
    parser.add_argument('--hr-tex-images', type=str, required=True)
    parser.add_argument('--lr-tex-images', type=str, required=True)
    parser.add_argument('--hr-occ-images', type=str, required=True)
    parser.add_argument('--lr-occ-images', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True) # 这里指示生成的h5文件的地址
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=100)
    # parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--eval', action='store_true') # 当'--eval'出现时，将其状态设为true
    
    # 从默认的地方获取数据，并写入默认的数据结构
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
