import numpy  as np
import tensorflow as  tf

default_padding='SAME'

def layer(op):
    def decorator(self,*args,**kwargs):

        name=kwargs['name']
        input=self.input[0]
        output=op(self,input,*args,**kwargs)
        print("output shape is:",output.get_shape())
        self.layers[name]=output
        self.feed(name)
        return self
    return decorator


class Network(object):
    def __init__(self,input):
        self.input=[]
        self.layers=dict(input)

        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed')

    def get_output(self,name):
        assert name in self.layers.keys(),"Keys Error"
        return self.layers[name]

    def feed(self,*args):
        assert len(args)!=0
        self.input=[]
        for i in args:
            self.input.append(self.layers[i])
        return self

    def mak_var(self,name,shape,initializer):
        return tf.get_variable(name=name,shape=shape,initializer=initializer)


    @layer
    def conv(self,input,k_w,k_h,c_o,s_w,s_h,name,relu=True,padding=default_padding):
        c_i=input.get_shape()[-1]

        conv=lambda i,w :tf.nn.conv2d(i,w,[1,s_w,s_h,1],padding=padding)

        with tf.variable_scope(name) as scope:
            init_weight=tf.truncated_normal_initializer(0.0,stddev=0.01)
            init_bias=tf.constant_initializer(0.1)
            weights=self.mak_var('weights',[k_w,k_h,c_i,c_o],init_weight)
            bias=self.mak_var('biases',[c_o],init_bias)

            if relu :
                return tf.nn.relu(conv(input,weights)+bias,name=scope.name)
            else:
                return tf.nn.bias_add(conv(input,weights),bias,name=scope.name)
    @layer
    def maxpooling(self,input,k_w,k_h,s_w,s_h,name,padding =default_padding):

        return tf.nn.max_pool(input,ksize=[1,k_w,k_h,1],strides=[1,s_w,s_h,1],padding =padding,name=name)


    @layer
    def reshape(self,input,name):
        print(input)
        print(input.shape[0])
        output=tf.reshape(input,[input.shape[0],-1],name=name)
        print(output.get_shape())
        return output

    @layer
    def fc(self,input,o_dim,name,relu=True):
        with tf.variable_scope(name) as scope:
            i_dim=input.get_shape()[-1]
            # print(input.get_shape())
            init_weight=tf.truncated_normal_initializer(0.0,stddev=0.01)
            init_bias=tf.constant_initializer(0.0)

            weights=self.mak_var('weights',[i_dim,o_dim],init_weight)
            bias=self.mak_var('biases',[o_dim],init_bias)

            op=tf.nn.relu_layer if relu else tf.nn.xw_plus_b

            return op(input,weights,bias,name=scope.name)

    @layer
    def softmax(self,input,name):
        return tf.nn.softmax(input,name=name)

    @layer
    def dropout(self,input,keep_pro,name):
        return tf.nn.dropout(input,keep_pro,name=name)
