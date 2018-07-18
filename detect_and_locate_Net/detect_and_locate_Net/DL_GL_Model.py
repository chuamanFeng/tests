import tensorflow as tf
import numpy as np
import basic_tf




#----------------------------------------------------serious model--------------------------------------------------#
def placeholder_inputs(batch_size,img_size,channel):

    img_pl=tf.placeholder(shape=[batch_size,img_size[0],img_size[1],channel],dtype=tf.float32)
    label_pl=tf.placeholder(shape=[batch_size,5],dtype=tf.int32)
    
    return img_pl,label_pl

def basic_detectModel(img,is_training,bn_decay,num_class):#512->4

    with tf.variable_scope('conv_unit1_G'):

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
    with tf.variable_scope('fully_connected_unit_G'):
               
        out2 = tf.reshape(out1,(int(out1._shape[0]),-1))  #b,8192
        #out2 = basic_tf.fully_connected(out2,4096,'fc1')
        #out2 = basic_tf.dropout(out2,is_training,'dp1',0.5)

        out2 = basic_tf.fully_connected(out2,1024,'fc2')
        out2 = basic_tf.dropout(out2,is_training,'dp2',0.5)

        out2 = basic_tf.fully_connected(out2,128,'fc3')
        out2 = basic_tf.dropout(out2,is_training,'dp3',0.5)
                

    with tf.variable_scope('output_unit_G'):

        pred = basic_tf.fully_connected(out2,(num_class+4),'fc4',activation_fn=None)
        
    return pred
  
def get_loss_G(pred,labels,scales,img_size,scope):

    with tf.variable_scope(scope):
        true_locate=tf.cast(labels[:,0:4],tf.float32)     
        true_class=tf.squeeze(labels[:,4:])

        pred_locate=pred[:,0:4]
        pred_class=pred[:,4:]

        coord_loss =tf.reduce_mean(tf.maximum(((true_locate[:,0]-pred_locate[:,0])/img_size[0])**2+((true_locate[:,1]-pred_locate[:,1])/img_size[1])**2,1e-10),-1)* scales[0]
        scale_loss =tf.reduce_mean(tf.maximum(((true_locate[:,2]-pred_locate[:,2])/img_size[0])**2+((true_locate[:,3]-pred_locate[:,3])/img_size[1])**2,1e-10),-1)* scales[1]   
        class_loss =tf.reduce_mean(tf.maximum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_class,logits=pred_class),1e-10))*scales[2]   
       
        loss=coord_loss+scale_loss+class_loss

    return loss

