import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


if __name__ == '__main__':
    #和参数的解析有关
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    #用于加速神经网络的训练
    cudnn.benchmark = True

    #用于指定tensor分配到哪个设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #用于指定model加载到某个设备
    model = SRCNN().to(device)
 
    state_dict = model.state_dict()#查看模型参数
    # torch.load(f, map_location=lambda storage, loc: storage)把f加载到GPU 1中，下述过程用于加载模型参数
    # items()以列表返回可遍历的(键, 值) 元组数组
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)#将文件中的参数应用于模型
        else:
            raise KeyError(n)

    #model测试样本时要在model(test_datasets)之前，加上model.eval(). 
    #否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。
    model.eval()
    
    #读取图像文件并转换为RGB
    image = pil_image.open(args.image_file).convert('RGB')

    #调整文件格式
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    #初始化数组并转换数据类型
    image = np.array(image).astype(np.float32)
    #将RGB转换成YCbCr
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)#from_numpy()完成从数组到tensor的转换
    y = y.unsqueeze(0).unsqueeze(0)#维度下标从0开始，这里表示两次给第一维增加维度的操作

    #torch.no_grad() 是一个上下文管理器
    #如果有不想被track的计算部分可以通过这么一个上下文管理器包裹起来。
    #这样可以执行计算，但该计算不会在反向传播中被记录。
    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)#torch.clamp(input, min, max, out=None)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))#format是一个用于格式化输出的函数

    #cpu()将处理数据从其他设备拿到cpu上
    #t.numpy()将Tensor变量转换为ndarray变量
    #其中t是一个Tensor变量，可以是标量，也可以是向量，转换后dtype与Tensor的dtype一致。
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    #transpose用于进行数据的转置
    #用np.transpose（img，(1,2,0)）
    #将图片的格式由（channels,imagesize,imagesize）转化为（imagesize,imagesize,channels）,
    #这样plt.show()就可以显示图片了
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])

    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)#用于实现从array到image的转换
    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))
