import tensorflow as tf

def main(_):
    logits = [[2,0.5,1],[0.1,1,3]]
    labels = [[0.2,0.3,0.5],[0.1,0.6,0.3]]
    logits_scaled = tf.nn.softmax(logits)

    result1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    result2 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)
    result3 = tf.nn.softmax_cross_entropy_with_logits(logits=logits_scaled,labels=labels)

    with tf.Session() as sess:
        print ('直接使用API计算softmax交叉熵：')
        print (sess.run(result1)  )
        print ('\n')
        print ('利用原理计算softmax交叉熵：')
        print (sess.run(result2))
        print ('\n')
        print (sess.run(logits_scaled))
        print ('错误！将logits输入时先用softmax缩放：')
        print (sess.run(result3))

if __name__ == '__main__':
    tf.app.run()
