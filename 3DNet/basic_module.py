import os
import sys
import tensorflow as tf
import numpy as np
import basic_tf
from PN_modules  import pointnet_AB_module,pointnet_AB_module_msg
import time

def abstraction_layer_sg(l0_xyz, l0_points,scope,is_training,bn_decay,tnet_spec=False):
     #Abstraction layers    
    with tf.variable_scope(scope):
        l1_xyz, l1_points,_= pointnet_AB_module(l0_xyz, l0_points,m=12800, r=1.6, ns=128, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1',tnet_spec=tnet_spec)
        l2_xyz, l2_points,_= pointnet_AB_module(l1_xyz, l1_points,m=2048, r=6.4, ns=128, mlp=[32,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2',tnet_spec=tnet_spec)
        l3_xyz, l3_points,_ = pointnet_AB_module(l2_xyz, l2_points, m=512, r=32.0, ns=64, mlp=[64,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3',tnet_spec=tnet_spec)       
        l4_xyz, l4_points,_ = pointnet_AB_module(l3_xyz, l3_points, m=None, r=None, ns=None, mlp=[128,256,512], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4')
      
    return [l1_xyz,l2_xyz, l3_xyz, l4_xyz] ,[l1_points, l2_points, l3_points, l4_points]

def abstraction_layer_mt(l0_xyz, l0_points,scope,is_training,bn_decay):

    with tf.variable_scope(scope):
        l1_xyz, l1_points = pointnet_AB_module_msg(l0_xyz, l0_points, 12800, [0.8,1.6], [32,64], [[32,32], [32,64]], is_training, bn_decay, scope='layer1')
        l2_xyz, l2_points = pointnet_AB_module_msg(l1_xyz, l1_points, 2048, [3.2,6.4], [64,64], [[32,64], [64,64]], is_training, bn_decay, scope='layer2')
        l3_xyz, l3_points = pointnet_AB_module_msg(l2_xyz, l2_points, 512, [16.0,32.0], [64,128], [[64,64],[64,128]], is_training, bn_decay, scope='layer3')    
        l4_xyz, l4_points, _ = pointnet_AB_module(l3_xyz, l3_points,m=None, r=None, ns=None, mlp=[128,256,512], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    return [l1_xyz,l2_xyz, l3_xyz, l4_xyz] ,[l1_points, l2_points, l3_points, l4_points]


def fullyconnected_layer(input,batch_size,class_num,scope,is_training,bn_decay):

    with tf.variable_scope(scope):  
        net = tf.reshape(input,(batch_size,-1))       
        net = basic_tf.fully_connected(net,512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = basic_tf.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
        net = basic_tf.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        net = basic_tf.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
        net = basic_tf.fully_connected(net,class_num, activation_fn=None, scope='fc3')

    return net