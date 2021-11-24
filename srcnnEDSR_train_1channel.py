import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 5"
import copy

import numpy as np
import torch
from torch import nn
import torch.optim as optim #一个实现了各种优化算法的库
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm #进度提示信息
from torch.utils.tensorboard import SummaryWriter #tensorlog输出，用于后续tensorboard监测

from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr
from srcnnEDSR_option import args
from srcnnEDSR_model import srcnnEDSR

kwargs={'map_location':lambda storage, loc: storage.cuda([0,1,2,3])}
def load_GPUS(model,model_path,kwargs):
    state_dict = torch.load(model_path,**kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k] = v
        print(k)
    # load params
    model.load_state_dict(new_state_dict)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default="/home/wangdanying/SRCNN/trainset_seq23GOF0")
    parser.add_argument('--eval-file', type=str, default="/home/wangdanying/SRCNN/evalset_seq23GOF0")
    parser.add_argument('--outputs-dir', type=str, default="/home/wangdanying/SRCNN/srcnnPytorch/debug/trainLog")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                        help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')                
    # 初始化模型需要的参数
    parser.add_argument('--num_resblocks', type=int, default=8)
    parser.add_argument('--num_feats', type=int, default=16)
    parser.add_argument('--num_colors', type=int, default=1)
    parser.add_argument('--scale', type=int, default=1)
    
    args = parser.parse_args()
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    cudnn.benchmark = True
    
    model = srcnnEDSR(args)
    model = nn.DataParallel(model)
    model = model.cuda()

    # 用于加载之前训练的参数
    # load_GPUS(model,"/home/wangdanying/SRCNN/srcnnPytorch/WDYbestparas/srcnnEDSR_res8_feats16_noUpsample_GOF0-3.pth",kwargs)

    torch.manual_seed(args.seed)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=args.batch_size,
                                  shuffle=True, # 是否打乱数据
                                  num_workers=args.num_workers, 
                                  pin_memory=True, # pin_memory为True，则data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中
                                  drop_last=True) # drop_last为True，则将最后不足batch size的数据丢弃，默认为false
   
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    best_weights = copy.deepcopy(model.state_dict()) # 使用deepcopy可以保证有list嵌套时，子list的修改之间是相互独立的
    best_epoch = 0
    best_psnr = 0.0
    writer = SummaryWriter('/home/wangdanying/SRCNN/srcnnPytorch/debug/tensorLog')
    
    for epoch in range(args.num_epochs):
        ################## 训练集 ###################
        # 设置为训练模式，train是模型训练的入口
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1)) # 用于输出信息
            # print("加载数据开始。。。。。")
            for data in train_dataloader:
                # print("加载数据")
                # 读取数据的时候是以元组的形式获得了低/高分辨率的图像
                inputs, labels = data
                # 多卡训练
                inputs = inputs.cuda()
                labels = labels.cuda()

                preds = model(inputs) 
                loss = criterion(preds, labels) 
                # 调用AverageMeter()中定义的update函数来更新损失的计算
                epoch_losses.update(loss.item(), len(inputs)) 

                # 计算梯度并进行更新
                optimizer.zero_grad() # 将梯度清零
                loss.backward() # 对tensor求导数，tensor是标量的时候不需要参数，这里loss是标量
                optimizer.step() # 使用参数空间的梯度来更新信息

                # 设置进度条后面显示的信息
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                # 设置手动更新，https://www.cnblogs.com/q735613050/p/10127531.html
                t.update(len(inputs))
        # 保存模型，.pth是python中的模型文件
        # print("保存模型。。。。。")
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
        
        ################## 验证集 ###################
        # model测试样本时要在model(test_datasets)之前，加上model.eval(). 
        # 否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。
        model.eval()

        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            # print("进入循环。。。。。")
            inputs, labels = data
            
            # 多卡训练
            inputs = inputs.cuda()
            labels = labels.cuda()

            # torch.no_grad() 是一个上下文管理器，该语句覆盖的部分将不会追踪梯度。
            # print("提取数据。。。。。")
            with torch.no_grad():
                # 这里的clamp是将数值限定在0-1的范围内
                preds = model(inputs).clamp(0.0, 1.0)
                # 下面的操作是去除空维度
                preds = np.squeeze(preds)
                labels = np.squeeze(labels)
                # print("preds.shape=",preds.shape)
                # print("labels.shape=",labels.shape)

                preds_valid = preds[:][0]
                labels_valid = labels[:][0]
                # print("preds[:][0].shape=", preds_valid.shape )
                # print("labels[:][0].shape=", labels_valid.shape )
                

            # 根据AverageMeter()中定义的update函数更新psnr相关的值
            # print("计算psnr。。。。。")
            epoch_psnr.update(calc_psnr(preds_valid, labels_valid), len(inputs))

        writer.add_scalar("srcnnEDSR_psnr_1epoch", epoch_psnr.avg, epoch)
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        # 权值更新
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    #保存最优权值
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
