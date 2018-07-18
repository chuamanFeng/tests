import tensorflow as tf
import numpy as np
import basic_tf
import sys
import os
import time


IMAGE_H=1024
IMAGE_W=1024
IMAGE_SIZE=[IMAGE_H,IMAGE_W]
SRGB2LAB=np.array([[0.142453,0.212671,0.019334,],
                    [0.357580,0.715160,0.119193],
                    [0.180432,0.072169,0.950227]]).astype('float32')
param1=7.787
param2=0.13793103
param3=0.33333333333

X0Y0Z0=np.array([94.285,
                 100.,
                 107.381]).astype('float32')
LEV=0.008856




def ssc_color_info_abstraction(input,mlp_list,is_training,bn_decay,scope,kernel_size,bn=True):
       
    kernel_h=kernel_size[0]
    kernel_w=kernel_size[1]
    channel=input.get_shape()[-1]
    with tf.variable_scope(scope) as sc:        
        each_layer=basic_tf.conv2d(input,channel, [1,1],
                                                padding='SAME', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv_p0', bn_decay=bn_decay)           
   
        for i,num_out_channel in enumerate(mlp_list):
            print('conv',i)
            input = basic_tf.conv2d(input,mlp_list[i]//2, [1,1],
                                  padding='SAME', stride=[1,1], bn=bn, is_training=is_training,
                                  scope='conv_p1%d'%(i), bn_decay=bn_decay)
            input = basic_tf.conv2d(input,mlp_list[i], [kernel_h,kernel_w],
                                    padding='SAME', stride=[1,1], bn=bn, is_training=is_training,
                                    scope='convp2_%d'%(i), bn_decay=bn_decay)   
                     
            each_layer=tf.concat([each_layer,input],-1)    #get a depth_concatened hypercolume (b,h1,w1,sum(mlp))
                        
        #pooling kernel-size[2,2]
       
        each_layer = basic_tf.max_pool2d(each_layer,kernel_size=[2,2],stride=[1,1], padding='SAME', scope='maxpool1')
     
            #(b,h2,w2,sum[mlp])
    print('one turn finished')
    return each_layer

     

def srgb2lab(data):

      
    b=int(data.shape[0])
    h=int(data.shape[1])
    w=int(data.shape[2])

   
    srgb2xyz=SRGB2LAB
   
    srgb2xyz=tf.tile(tf.reshape(srgb2xyz,(1,1,1,3,3)),[b,h,w,1,1])
    data=tf.tile(tf.expand_dims(data,-1),[1,1,1,1,3])
    data=tf.cast(data,tf.float32)

    XYZ=tf.reshape(tf.reduce_sum(tf.multiply(srgb2xyz,data),3),(b,h,w,3))
    x0y0z0=tf.tile(tf.reshape(X0Y0Z0,(1,1,1,3)),[b,h,w,1])
    xyz=tf.truediv(XYZ,x0y0z0)

    
    premask=tf.zeros_like(xyz,tf.float32)
    xyzsm=tf.minimum((xyz-LEV),0)   #(<LEV)
    xyzbi=tf.maximum((xyz-LEV),0)   #(>LEV)
  
    
    masksm=tf.cast(tf.equal(xyzsm,premask),tf.float32)
    maskbi=tf.cast(tf.equal(xyzbi,premask),tf.float32)
    
    xyz1=xyz*param1+param2
    xyz2=xyz**param3
  
    xyzsm=tf.multiply(masksm,xyz1)
    xyzbi=tf.multiply(maskbi,xyz2)
    
    Lab=tf.add(xyzsm,xyzbi)
  
    return tf.reshape(Lab,(b,h*w,3))

#if __name__=='__main__':

