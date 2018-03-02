import numpy as np
from  LeNet_test import  LeNet_test
from get_db import get_db
import pandas as pd
import tensorflow as tf
import argparse
import sys
from pandas import DataFrame


def parse_args():
    """
    parse input arguments
    """
    parser=argparse.ArgumentParser(description='Test LeNet for minst')

    parser.add_argument('--filename',dest='filename',
                        help='获取文件的位置')
    parser.add_argument('--output_dir',dest='output_dir',
                        help='获取权重的位置',
                        default='lenet_weight/train_weights_10000_iter.ckpt')
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args=parser.parse_args()

    return args

if __name__=='__main__':
    args=parse_args()
    filename=args.filename
    output_dir=args.output_dir
    db=get_db(filename)

    lenet_test=LeNet_test()

    #cls_pro
    cls_pro=lenet_test.get_output('cls_pro')
    index=tf.argmax(cls_pro,1)
    saver=tf.train.Saver()
    test=np.zeros((db.shape[0],2),dtype=np.int)
    print("Predicting")
    with tf.Session() as sess:
        saver.restore(sess,output_dir)
        for  i in range(db.shape[0]):
            feed_dict={lenet_test.x:db[i,:].reshape((-1,28,28,1))}
            # index1=sess.run([index],feed_dict=feed_dict)
            index1=index.eval(feed_dict=feed_dict)
            test[i,0]=i+1
            test[i,1]=index1
            print(index1)
    print("Ending")

    df=DataFrame(test)
    df.to_csv('result1.csv',header={'ImageId','Label'},index=False,sep=',')
