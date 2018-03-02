from network import Network
import numpy as np
import tensorflow as tf

class LeNet_test(Network):
    def __init__(self):
        self.x=tf.placeholder(tf.float32,shape=[1,28,28,1])
        self.input=[]
        self.layers=dict({'x':self.x})
        # self.keep_pro=keep_pro
        self.setup()

    def setup(self):
        (self.feed('x')
             .conv(5,5,32,1,1,name='C1')
             .maxpooling(2,2,2,2,name='S2',padding='VALID')
             .conv(5,5,64,1,1,name='C3')
             .maxpooling(2,2,2,2,name='S4',padding='VALID')
             .reshape(name='r')
             .fc(1024,name='fc6')
            #  .dropout(keep_pro=self.keep_pro,name='drop6')
             .fc(10,name='fc7',relu=False)
             .softmax(name='cls_pro'))
