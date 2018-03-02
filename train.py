from LeNet_train import LeNet_train
from get_db import get_db
import numpy as np
import os
import tensorflow as tf

class SolverWrapper(object):
    def __init__(self,net,sess,saver,data,labels,
                 output_dir,sample_batch,max_iter,snap_shot_gap):

        self.net=net
        self.saver=saver
        #记录所有的数据，存储起来，不知道FasterRCNN时怎么做的
        self.labels=labels
        self.sess=sess
        self.data=data
        self.data_number=data.shape[0]
        self.output_dir=output_dir
        self.sample_batch=sample_batch
        self.max_iter=max_iter
        self.snap_shot_gap=snap_shot_gap

    #每次训练过程中我们要随机抽取一些数据
    #返回随机抽取的x,labels
    def sample_data(self):
        index=np.random.choice(self.data_number,self.sample_batch,replace=False)

        x=self.data[index,:,:]
        y=self.labels[index]
        return x,y

    #每多少次迭代我们需要记录我们的参数
    #这时候我们需要用tensorflow自带的
    def snap_shot(self,iter):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename='train_weights_'+'{:d}_iter'.format(iter+1)+'.ckpt'
        output_dir=os.path.join(self.output_dir,filename)
        self.saver.save(self.sess,output_dir)
        print('Wrote snapshot to {:s}'.format(output_dir))

    def train_model(self):
        #定义损失函数
        #首先获取网络中的scores
        #然后与labels做对比

        labels=self.net.labels #sample_batch*1
        scores=self.net.get_output('cls_pro')#sample_batch*10
        class_num=self.net.class_num
        labels=tf.expand_dims(labels,1)
        labels=tf.cast(labels,tf.int32)
        indices=tf.expand_dims(tf.range(0,self.sample_batch,1),1)
        concated=tf.concat([indices,labels],1)
        onehot_labels=tf.sparse_to_dense(concated,
                                        tf.stack([self.sample_batch,class_num]),1.0,0.0)
        #-y_ilog(prob)

        cross_entropy_loss=tf.reduce_mean(tf.reduce_sum(-tf.log(scores)*onehot_labels,
                                            1))

        train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)

        self.sess.run(tf.global_variables_initializer())
        for i in range(self.max_iter):
            sample_x,sample_y=self.sample_data()
            feed_dict={self.net.x:sample_x,self.net.labels:sample_y}
            _,loss=self.sess.run((train_step,cross_entropy_loss),feed_dict=feed_dict)

            if (i+1)%self.snap_shot_gap==0:
                self.snap_shot(i)

            if i%100==0:
                print('{:d} iter＇s loss　is :{:f}'.format(i,loss))


def train(filename,output_dir,sample_batch=64,max_iter=20000,snap_shot_gap=1000,keep_pro=0.5,class_num=10):
    data,labels=get_db(filename)

    #sess,saver,net
    net=LeNet_train(keep_pro,class_num)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        solver=SolverWrapper(net,sess,saver,data,labels,output_dir,
                            sample_batch,max_iter,snap_shot_gap)
        print("Solving")
        solver.train_model()
        print("End solving")
