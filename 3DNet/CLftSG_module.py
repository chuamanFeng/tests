import os
import sys
import tensorflow as tf
import numpy as np
import basic_tf
from PN_modules import pointnet_AB_module,pointnet_AB_module_msg, pointnet_fp_module
import time

#------------------------------------------------------------single-as-method-----------------------------------------------------------------#
def get_model_single_seg(point_cloud, is_training, bn_decay=None,is_tnet=False):
   
    b = point_cloud.get_shape()[0].value
    n = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = point_cloud

    if is_tnet:
        tnet_spec={'mlp':[64,128,1024], 'is_training':is_training, 'bn_decay':bn_decay}
    else:
        tnet_spec=None

    end_points['l0_xyz'] =l0_xyz 

    #Abstraction layers 
    l1_xyz, l1_points,_ = pointnet_AB_module(l0_xyz, l0_points, m=51200, r=0.4, ns=64, mlp=[32,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1',tnet_spec=tnet_spec)    
    l2_xyz, l2_points,_= pointnet_AB_module(l1_xyz, l1_points,m=12800, r=1.6, ns=128, mlp=[64,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2',tnet_spec=tnet_spec)
    l3_xyz, l3_points,_ = pointnet_AB_module(l2_xyz, l2_points, m=5120, r=3.2, ns=64, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3',tnet_spec=tnet_spec)    
    l4_xyz, l4_points,_= pointnet_AB_module( l3_xyz, l3_points,m=2048, r=6.4, ns=64, mlp=[256,512,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4',tnet_spec=tnet_spec)
    l5_xyz, l5_points,_ = pointnet_AB_module(l4_xyz, l4_points, m=1024, r=19.2, ns=64, mlp=[512,1024,1024], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer5',tnet_spec=tnet_spec)    
    l6_xyz, l6_points,_= pointnet_AB_module(l5_xyz, l5_points,m=512, r=38.4, ns=64, mlp=[1024,1024,2048], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer6',tnet_spec=tnet_spec)
    l7_xyz, l7_points,_ = pointnet_AB_module(l6_xyz, l6_points, m=None, r=None, ns=None, mlp=[2048,2048,4096], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer7')
      
    # Fully connected layers
    netc = tf.reshape(l7_points, [b,-1])   
    netc = basic_tf.fully_connected(netc,2048, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    netc = basic_tf.dropout(netc, keep_prob=0.8, is_training=is_training, scope='dp1')
    netc = basic_tf.fully_connected(netc,1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    netc = basic_tf.dropout(netc, keep_prob=0.8, is_training=is_training, scope='dp2')
    netc = basic_tf.fully_connected(netc, 512, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
    netc = basic_tf.dropout(netc, keep_prob=0.5, is_training=is_training, scope='dp3')
    netc = basic_tf.fully_connected(netc, 128, bn=True, is_training=is_training, scope='fc4', bn_decay=bn_decay)
    netc = basic_tf.dropout(netc, keep_prob=0.5, is_training=is_training, scope='dp4')
    netc = basic_tf.fully_connected(netc, 40, activation_fn=None, scope='fc5')


    # Feature Propagation layers&fully connected layers for segmentation
    l6_points = pointnet_fp_module(l6_xyz, l7_xyz, l6_points, l7_points, [2048,2048,1024], is_training, bn_decay, scope='fa_layer1') 
    l5_points = pointnet_fp_module(l5_xyz, l6_xyz, l5_points, l6_points, [1024,1024], is_training, bn_decay, scope='fa_layer2')
    l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4_points, l5_points, [1024,512], is_training, bn_decay, scope='fa_layer3') 
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512,512], is_training, bn_decay, scope='fa_layer4')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512,256], is_training, bn_decay, scope='fa_layer5') 
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer6')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer7') #(32,102400,128)
   
    net = basic_tf.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fcs1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = basic_tf.dropout(net, keep_prob=0.5, is_training=is_training, scope='dps1')
    net = basic_tf.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fcs2')


    return netc,net,end_points
#------------------------------------------------------------multiple-as-method-----------------------------------------------------------------#
def get_model_multi_seg(point_cloud, is_training, bn_decay=None):
    
    b = point_cloud.get_shape()[0].value
    n = point_cloud.get_shape()[1].value
    c=point_cloud.get_shape()[2].value
    end_points = {}

    l0_points = point_cloud    
    if c>3:
        l0_xyz = tf.slice(point_cloud,[0,0,0],[-1,-1,3])
    else:
        l0_xyz = point_cloud
   

     #Abstraction layers
    l1_xyz, l1_points = pointnet_AB_module_msg(l0_xyz, l0_points, 51200, [0.2,0.4,0.8], [32,64,96], [[32,32,64], [64,64,128], [64,96,128]], is_training, bn_decay, scope='layer1')
    l2_xyz, l2_points = pointnet_AB_module_msg(l1_xyz, l1_points, 12800, [0.8,1.6,3.2], [128,128,256], [[64,64,128], [128,128,256], [128,128,256]], is_training, bn_decay, scope='layer2')
    l3_xyz, l3_points = pointnet_AB_module_msg(l2_xyz, l2_points, 5120, [1.6,3.2,6.4], [64,128,128], [[128,128,256], [256,256,512], [512,512,1024]], is_training, bn_decay, scope='layer3')
    l4_xyz, l4_points = pointnet_AB_module_msg(l3_xyz, l3_points,2048, [3.2,6.4,12.8], [128,128,256], [[256,256,512], [512,512,1024], [1024,1024,2048]], is_training, bn_decay, scope='layer4')
    l5_xyz, l5_points = pointnet_AB_module_msg(l4_xyz, l4_points,1024, [12.8,25.6], [64,128], [[512,1024,1024], [1024,2048,2048]], is_training, bn_decay, scope='layer5')
    l6_xyz, l6_points = pointnet_AB_module_msg(l5_xyz, l5_points,512, [25.6,51.2], [32,64], [[512,1024,1024], [1024,2048,2048]], is_training, bn_decay, scope='layer6')
    l7_xyz, l7_points, _ = pointnet_AB_module(l6_xyz, l6_points,m=None, r=None, ns=None, mlp=[2048,2048,4096], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer7')
  
    # Fully connected layers for classification
    netc = tf.reshape(l7_points, [b, -1])
    netc = basic_tf.fully_connected(netc,2048, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    netc = basic_tf.dropout(netc, keep_prob=0.8, is_training=is_training, scope='dp1')
    netc = basic_tf.fully_connected(netc, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    netc = basic_tf.dropout(netc, keep_prob=0.8, is_training=is_training, scope='dp2')
    netc = basic_tf.fully_connected(netc, 512, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
    netc = basic_tf.dropout(netc, keep_prob=0.5, is_training=is_training, scope='dp3')
    netc = basic_tf.fully_connected(netc, 128, bn=True, is_training=is_training, scope='fc4', bn_decay=bn_decay)
    netc = basic_tf.dropout(netc, keep_prob=0.5, is_training=is_training, scope='dp4')
    netc = basic_tf.fully_connected(netc, 40, activation_fn=None, scope='fc5')

     # Feature Propagation layers&fully connected layers for segmentation
    l6_points = pointnet_fp_module(l6_xyz, l7_xyz, l6_points, l7_points, [2048,2048,1024], is_training, bn_decay, scope='fa_layer1') 
    l5_points = pointnet_fp_module(l5_xyz, l6_xyz, l5_points, l6_points, [1024,1024], is_training, bn_decay, scope='fa_layer2')
    l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4_points, l5_points, [1024,512], is_training, bn_decay, scope='fa_layer3') 
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512,512], is_training, bn_decay, scope='fa_layer4')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512,256], is_training, bn_decay, scope='fa_layer5') 
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,256], is_training, bn_decay, scope='fa_layer6')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [256,128,128], is_training, bn_decay, scope='fa_layer7') #(32,102400,128)
        
    net = basic_tf.conv1d(l0_points,128, 1, padding='VALID', bn=True, is_training=is_training, scope='fcs1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = basic_tf.dropout(net, keep_prob=0.5, is_training=is_training, scope='dps1')
    net = basic_tf.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fcs3')

    return netc,net,end_points

#------------------------------------------------------------double-loss-----------------------------------------------------------------#
def get_loss(pred1, label1, pred2, label2,weight=None):
    """ pred: BxNxC,
        label: BxN,
        weight:a dict with 2 weights{keys=w1,w2} """
    loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label1,logits=pred1) #(b,n)
    loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label2,logits=pred2) #(b,n)

    w1=weight['w1']
    w2=weight['w2']

    if w1+w2!=1.:
        w1=0.7
        w2=0.3       
   

    classify_loss = w1*tf.reduce_mean(loss1)   #(1)
    segment_loss = w2*tf.reduce_mean(loss2)
    double_loss=classify_loss+segment_loss

    tf.summary.scalar('double loss', double_loss)
    return double_loss

with tf.Graph().as_default():
        now =time.time() 
        pts = np.random.random((32,102498,6)).astype('float32')
        inputs = tf.constant(pts)
        output,output2, _ = get_model_multi_seg(inputs, tf.constant(True))        #62.3sec
       # output,output2, _ = get_model_single_seg(inputs, tf.constant(True),is_tnet=True)      # 60sec(True),
        print(time.time()-now)
        print(output)
        print(output2)