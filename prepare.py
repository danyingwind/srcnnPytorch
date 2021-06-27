import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image# python的一个图像处理模块
from utils import convert_rgb_to_y

pil_image.LOAD_TRUNCATED_IMAGES = True

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
    # 下面sorted()的意思是对以args.images_dir为前缀的所有路径进行排序，for是对每张图进行遍历
    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        hr = pil_image.open(image_path).convert('RGB')
        #如果不用convert('RGB')，会导致图像是4通道RGBA，A是透明度通道，现在用不到

        # 尺寸调整
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale

        # BICUBIC获取高/低分辨率的图像
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)

        # 数据类型转换
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        # 根据RGB求Y
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        # CNN处理图像的时候，不是一次输入一整块，而是把图像划分成多个patch
        # 获取每张图的lr_patches和hr_patches
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride): 
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

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

    # enumerate是枚举函数，i、image_path分别对应索引下标和路径
    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
       
        # 调整图像尺寸
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        # 获得低/高分辨率图像
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        # 数据格式转换
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        # 计算y分量
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

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
    parser.add_argument('--images-dir', type=str, required=True) # 这里train和eval都用了images_dir，不是同一个
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=33)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--eval', action='store_true') # 当'--eval'出现时，将其状态设为true
    
    # 从默认的地方获取数据，并写入默认的数据结构
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
