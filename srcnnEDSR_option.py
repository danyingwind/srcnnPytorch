import argparse
import os
# 用来设置训练时的参数
parser = argparse.ArgumentParser()
parser.add_argument('--train-file', type=str, default="/home/wangdanying/SRCNN/trainset_seq23GOF0")
parser.add_argument('--eval-file', type=str, default="/home/wangdanying/SRCNN/evalset_seq23GOF0")
parser.add_argument('--outputs-dir', type=str, default="/home/wangdanying/SRCNN/srcnnPytorch/debug/trainLog")
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-epochs', type=int, default=400)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--num_resblocks', type=int, default=32)
parser.add_argument('--num_feats', type=int, default=64)
parser.add_argument('--num_colors', type=int, default=2)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')                    
args = parser.parse_args()
# 生成outputs_dir的路径名
args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
#判断文件是否存在，不存在则创建一个
if not os.path.exists(args.outputs_dir):
    os.makedirs(args.outputs_dir)