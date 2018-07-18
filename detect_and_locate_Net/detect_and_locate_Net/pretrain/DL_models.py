import tensorflow as tf
import numpy as np
import basic_tf


#----------------------------------------------------pre-train model----------------------------------------------#
def placeholder_inputs_pre(batch_size,img_size,channel):

    img_pl=tf.placeholder(shape=[batch_size,img_size[0],img_size[1],channel],dtype=tf.float32)
    label_pl=tf.placeholder(shape=[batch_size],dtype=tf.int32)
   
    return img_pl,label_pl

def brute_classify(img,num_class,is_training,bn_decay):#512->4

    with tf.variable_scope('conv_unit1'):

        out = basic_tf.conv2d(img,16,[3,3],'conv_11',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_11',[2,2],'SAME')

        out = basic_tf.conv2d(out,32,[3,3],'conv_12',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_12',[2,2],'SAME')

        out = basic_tf.conv2d(out,64,[3,3],'conv_13',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_13',[2,2],'SAME')
  
        out = basic_tf.conv2d(out,128,[3,3],'conv_14',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_14',[2,2],'SAME')

        out = basic_tf.conv2d(out,256,[3,3],'conv_15',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_15',[2,2],'SAME')

        out = basic_tf.conv2d(out,512,[3,3],'conv_16',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_16',[2,2],'SAME')

        out = basic_tf.max_pool2d(out,[2,2],'maxpool_17',[2,2],'SAME')
    with tf.variable_scope('conv_unit2'):

        out1 = basic_tf.conv2d(out,1024,[3,3],'conv_21',[1,1],'SAME')       
        out1 = basic_tf.conv2d(out1,512,[1,1],'conv_22',[1,1],'SAME')
        out1 = basic_tf.avg_pool2d(out1,[2,2],'pre_avepool',[2,2],'SAME')
        
    with tf.variable_scope('fully_connected_unit'):

        out2 = tf.reshape(out1,(int(out1._shape[0]),-1))  #b,4096     
        out2 = basic_tf.fully_connected(out2,1024,'fc1')
        out2 = basic_tf.dropout(out2,is_training,'dp1',0.5)

        out2 = basic_tf.fully_connected(out2,128,'fc2')
        out2 = basic_tf.dropout(out2,is_training,'dp2',0.5)

        pred = basic_tf.fully_connected(out2,num_class,'fc3')
    print(pred)
    return pred

def get_loss_pre(predicts, labels):

    with tf.variable_scope('get_loss'):
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predicts,labels=labels)
        loss=tf.reduce_mean(loss,0)
    return loss



if __name__=='__main__':
  
    img=np.random.rand(4,224,224,3).astype('float32')   
    labels=np.random.rand(4,1,5).astype('int32')
   
    img=tf.constant(img)
    labels=tf.constant(labels)
    cell_size=[4,4]
    image_size=512
    class_num=2
    box_num=2
    #pre=brute_classify(img,tf.constant(True),None,class_num)
    pre=basic_detectModel(img,tf.constant(True),None,cell_size,class_num,box_num)
    res=get_loss(pre,labels,4,512,cell_size[0],class_num,box_num,[0.2,0.6,0.1,0.1])

    print(pre)
  