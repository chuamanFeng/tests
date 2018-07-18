import os
import sys
import tensorflow as tf
import numpy as np
import basic_tf
from PN_modules  import pointnet_AB_module,pointnet_AB_module_msg
import time
import basic_module as bm

def placeholder_inputs(b, n):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(b, n, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(b))
    return pointclouds_pl, labels_pl
#-----------------------------------------------------singleF-classification----------------------------------------------------------#
def get_model_single_cl(point_cloud,class_num,batch_size,is_training,bn_decay=None,is_tnet=False):
 
    n  =point_cloud.get_shape()[1].value
    c =point_cloud.get_shape()[2].value
    slice_size=n//3
    end_points = {}

    l0_points = point_cloud    
    with tf.variable_scope('input_layer_S'):
       
        if c>3:
            l0_xyz = l0_points[:,:,:3]
            l0_points=tf.concat((l0_xyz,l0_points[:,:,3:]/255.),-1)
        else:
            l0_xyz = point_cloud        

        end_points['l0_xyz'] = l0_xyz
        if is_tnet:
            tnet_spec={'mlp':[64,128,1024], 'is_training':is_training, 'bn_decay':bn_decay}
        else:
            tnet_spec=None

    with tf.variable_scope('interminate_layer_S'):  
        l0_xyz1,l0_points1=l0_xyz[:,0:slice_size,:],l0_points[:,0:slice_size,:]
        l0_xyz2,l0_points2=l0_xyz[:,slice_size:2*slice_size,:],l0_points[:,slice_size:2*slice_size,:]
        l0_xyz3,l0_points3=l0_xyz[:,2*slice_size:,:],l0_points[:,2*slice_size:,:]

        _,out1=bm.abstraction_layer_sg(l0_xyz1,l0_points1,'part1',is_training,bn_decay,tnet_spec)
        _,out2=bm.abstraction_layer_sg(l0_xyz2,l0_points2,'part2',is_training,bn_decay,tnet_spec)
        _,out3=bm.abstraction_layer_sg(l0_xyz3,l0_points3,'part3',is_training,bn_decay,tnet_spec)
        
    with tf.variable_scope('output_layer_S') as sc: 
        
        out1=tf.reshape(out1[-1],(batch_size,-1))
        out2=tf.reshape(out2[-1],(batch_size,-1))
        out3=tf.reshape(out3[-1],(batch_size,-1))

        out=tf.concat((out1,out2,out3),-1)
        pred=bm.fullyconnected_layer(out,batch_size,class_num,sc,is_training,bn_decay)
    

    return pred, end_points

#-----------------------------------------------------multipleF-classification----------------------------------------------------------#
def get_model_multi_cl(point_cloud,class_num,batch_size,is_training,bn_decay=None):
   
    n  =point_cloud.get_shape()[1].value
    c =point_cloud.get_shape()[2].value
    slice_size=n//3
    end_points = {}

    l0_points = point_cloud    
    with tf.variable_scope('input_layer_M'):
        if c>3:
            l0_xyz = l0_points[:,:,:3]
            l0_points=tf.concat((l0_xyz,l0_points[:,:,3:]/255.),-1)
        else:
            l0_xyz = point_cloud
        end_points['l0_xyz'] = l0_xyz

    with tf.variable_scope('interminate_layer_M'):  
        l0_xyz1,l0_points1=l0_xyz[:,0:slice_size,:],l0_points[:,0:slice_size,:]
        l0_xyz2,l0_points2=l0_xyz[:,slice_size:2*slice_size,:],l0_points[:,slice_size:2*slice_size,:]
        l0_xyz3,l0_points3=l0_xyz[:,2*slice_size:,:],l0_points[:,2*slice_size:,:]

        _,out1=bm.abstraction_layer_mt(l0_xyz1,l0_points1,'part1',is_training,bn_decay)
        _,out2=bm.abstraction_layer_mt(l0_xyz2,l0_points2,'part2',is_training,bn_decay)
        _,out3=bm.abstraction_layer_mt(l0_xyz3,l0_points3,'part3',is_training,bn_decay)
        
    # Fully connected layers
    with tf.variable_scope('output_layer_M') as sc: 

        out1=tf.reshape(out1[-1],(batch_size,-1))
        out2=tf.reshape(out2[-1],(batch_size,-1))
        out3=tf.reshape(out3[-1],(batch_size,-1))

        out=tf.concat((out1,out2,out3),-1)
        print(out.shape)
        pred=bm.fullyconnected_layer(out,batch_size,class_num,sc,is_training,bn_decay)
    
    
    

    return pred, end_points

#-----------------------------------------------------loss----------------------------------------------------------#
def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss


#with tf.Graph().as_default():
#        now =time.time() 
#        pts = np.random.random((32,102400,6)).astype('float32')
#        inputs = tf.constant(pts)
#        output, _ = get_model_multi_cl(inputs,3,32,tf.constant(True))     #40sc(6layers) with 102456 points,and 59sec(7layers) in 10.20
#        #output, _ = get_model_single_cl(inputs, tf.constant(True),is_tnet=True)     #58.4sec(False)
#        print(time.time()-now)
#        print(output)