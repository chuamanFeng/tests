import tensorflow as tf
import numpy as np
import sys
import os
import basic_tf

def input_transform_net(xyz, mlp,is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        mlp:list of output_channels
        Return:           
            transformed_xyz (b,n,3)"""
    b = xyz.get_shape()[0].value
    n =  xyz.get_shape()[1].value   
    print('transing')
    input= tf.expand_dims(xyz, -1)#(b,n,3,1)
    for i, num_out_channel in enumerate(mlp):
        if i==0:
             net = basic_tf.conv2d(input,num_out_channel, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv%d'%(i+1), bn_decay=bn_decay)
        else:
            net = basic_tf.conv2d(net, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn_decay, is_training=is_training,
                                         scope='tconv%d'%(i+1), bn_decay=bn_decay) #(b,n,mlp[-1])
  
    net = basic_tf.max_pool2d(net, [n,1],
                             padding='VALID', scope='tmaxpool')#(b,1,mlp[-1])

    net = tf.reshape(net, [b, -1])                           #(b * mlp[-1])
    net = basic_tf.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)#(b,512)
    net = basic_tf.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)#(b,256)
    print('transing2')
    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)  
        transform = tf.nn.bias_add(transform, biases)  #now,net_shape:(b,3*K)

    transform = tf.reshape(transform, [b, 3, K])   
    transformed_xyz=tf.matmul(xyz,transform)    
    return transformed_xyz

def feature_transform_net(inputs, mlp,is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        mlp:list of output_channels
        Return:
             transformed_inputs (b,n,k)"""
    b = xyz.get_shape()[0].value
    n =  xyz.get_shape()[1].value   

    net=inputs
    for i, num_out_channel in enumerate(mlp):      
          net = basic_tf.conv2d(net, num_out_channel, [1,1],
                                       padding='VALID', stride=[1,1],
                                       bn=bn, is_training=is_training,
                                       scope='tconv%d'%(i+1), bn_decay=bn_decay) 
   
    net = basic_tf.max_pool2d(net, [n,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [b, -1])
    net = basic_tf.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = basic_tf.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [b, K, K])
    transformed_inputs=tf.matmul(input,transform)
    return  transformed_inputs