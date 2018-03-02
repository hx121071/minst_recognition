from train import train
import argparse
import numpy as np
import sys

def parse_args():
    """
    parse input arguments
    """

    parser=argparse.ArgumentParser(description='Train LeNet for  minst recognition')

    parser.add_argument('--filename',dest='filename',
                        help='获取文件的位置')
    parser.add_argument('--output_dir',dest='output_dir',
                        help='权重保存的位置',default='lenet_weight')

    parser.add_argument('--sample_batch',dest='sample_batch',
                        help='每次随机抽取多少个',default=64)
    parser.add_argument('--max_iter',dest='max_iter',
                        help='训练的迭代次数',default=10000)
    parser.add_argument('--snap_shot_gap',dest='snap_shot_gap',
                        help='每训练多少次记录一次参数',default=1000)

    parser.add_argument('--keep_pro',dest='keep_pro',
                        help='激活概率',default=0.5)

    parser.add_argument('--class_num',dest='class_num',
                        help='分类数',default=10)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args=parser.parse_args()

    return args

if __name__=='__main__':
    args=parse_args()
    filename=args.filename
    output_dir=args.output_dir
    sample_batch=args.sample_batch
    max_iter=args.max_iter
    snap_shot_gap=args.snap_shot_gap
    keep_pro=args.keep_pro
    class_num=args.class_num
    print(filename,output_dir,sample_batch,max_iter,snap_shot_gap,keep_pro,class_num)
    train(filename,output_dir,sample_batch,max_iter,snap_shot_gap,keep_pro,class_num)
