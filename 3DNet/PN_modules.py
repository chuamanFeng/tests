import os
import sys
from sampling3d import farthest_point_sample, gather_point
from grouping3d import query_ball_point, group_point, knn_point
from interpolation3d import three_nn, three_interpolate, get_weight
import tensorflow as tf
import numpy as np
import basic_tf
from transnet import input_transform_net

#---------------------------------------------------preparations-----------------------------------------------------#

def placeholder_inputs(b, n):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(b, n, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(b))
    return pointclouds_pl, labels_pl
#******************************transformation matirx for cloud-coordination********************************#
def tnet(grouped_xyz, tnet_spec):
    '''
    Input:
        tnet_spec :dict (keys: mlp,is_training, bn_decay)
        grouped_xyz (b,ns,3)
    Output:
        transformed_xyz(b,ns,3)
    '''
    mlp=tnet_spec['mlp']   
    is_training=tnet_spec['is_training']
    bn_decay=tnet_spec['bn_decay']

    if mlp!=None:
        transformed_xyz=input_transform_net(grouped_xyz,mlp,is_training,bn_decay)
    else:
        transformed_xyz=grouped_xyz
  
    return transformed_xyz

#******************************sampling&grouping********************************#

def sample_and_group(m, r, ns, xyz, points, tnet_spec=None, knn=False, use_xyz=True):
    '''
    Input:
        m: int32
        r: float32
        nse: int32
        xyz: (b, n, 3) TF tensor
        points: (b, n, c) TF tensor
        tnet_spec: dict (keys: mlp,is_training, bn_decay), if None do not apply tnet
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (b, m, 3) TF tensor
        new_points: (b, m, ns, 3+c) TF tensor
        idx: (b, m, ns) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, m, ns, 3) TF tensor, normalized point XYZs
            (subtracted by center point XYZ) in local regions
    '''  
    new_xyz = gather_point(xyz, farthest_point_sample(m, xyz)) # (b, n, c)
    b=new_xyz.get_shape()[0].value
    if knn:
        _,idx = knn_point(ns, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(r, ns, xyz, new_xyz)   #some problem in new_xyz           
    grouped_xyz = group_point(xyz, idx) # (b, m, ns, 3)   
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,ns,1]) # move away the center-points themselves,norm the origin
   
    if tnet_spec is not None:
        grouped_xyz=tf.reshape(grouped_xyz,(b*m,ns,3))       
        grouped_xyz = tf.reshape(tnet(grouped_xyz, tnet_spec),(b,m,ns,3))  #(b*m,ns,3)  no problem
    if points is not None:     
        grouped_points = group_point(points, idx) # (b, m, ns, c)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (b, m, ns, 3+c)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

#******************************transformation matirx for cloud-coordination********************************#

def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (b, n, 3) TF tensor
        points: (b, n, c) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (b, 1, 3) as (0,0,0)
        new_points: (b, 1, n, 3+c) TF tensor
    Note:
        Equivalent to sample_and_group with m=1, r=inf, use (0,0,0) as the centroid
    '''
    b = xyz.get_shape()[0].value
    ns = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (b,1,1)),dtype=tf.float32) # (b, 1, 3)center*b��[[0,0,0]]*b
    idx = tf.constant(np.tile(np.array(range(ns)).reshape((1,1,ns)), (b,1,1)))
    grouped_xyz = tf.reshape(xyz, (b, 1, ns, 3)) # (b,m=1, ns, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (b, 16, 256+3=259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (b, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz

#---------------------------------------------------feature-net-module-----------------------------------------------------#

#*****************************singleRadius-multiLayer-feature-Abstraction****************************************#

def pointnet_AB_module(xyz, points,m, r, ns, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', tnet_spec=None, knn=False, use_xyz=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (b, n, 3) TF tensor 
            points: (b, n, c) TF tensor 
            m: int32 -- #points sampled in farthest point sampling 
            r: float32 -- search radius in local region
            ns: int32 -- how many points in each local region 
            mlp: list of int32 -- output size for MLP on each point 
            mlp2: list of int32 -- output size for MLP on each region 
            group_all: bool -- group all points into one PC if set true, OVERRIDE   
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features 
        Return:
            new_xyz: (b,m, 3) TF tensor
            new_points: (b,m, mlp[-1] or mlp2[-1]) TF tensor
            idx: (b,m, ns) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope) as sc:

        #Sampling&Grouping

        if group_all:
            ns = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(m, r, ns, xyz, points, tnet_spec, knn, use_xyz)#here we got the idx from sampling&grouping
        print('convolution')
        #convolutional layer mlp(handling the new_points we got)

        for i, num_out_channel in enumerate(mlp):
            print('conv',i)
            new_points = basic_tf.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay) 
            
        #pooling
        print('pooling')
        if pooling=='avg':
            new_points = basic_tf.avg_pool2d(new_points, [1,ns], stride=[1,1], padding='VALID', scope='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (b, m, ns, 1)
                new_points *= weights # (b, m, ns, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        elif pooling=='min':
            new_points = basic_tf.max_pool2d(-1*new_points, [1,ns], stride=[1,1], padding='VALID', scope='minpool1')
        elif pooling=='max_and_avg':
            avg_points = basic_tf.max_pool2d(new_points, [1,ns], stride=[1,1], padding='VALID', scope='maxpool1')
            max_points = basic_tf.avg_pool2d(new_points, [1,ns], stride=[1,1], padding='VALID', scope='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        #convolutional layer mlp2

        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = basic_tf.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay) 

        #prepare the result

        new_points = tf.squeeze(new_points, [2]) # (b,m,mlp2[-1])
        print('1 turn')
        return new_xyz, new_points, idx

#*****************************multiRadius-multiLayer-feature-Abstraction****************************************#

def pointnet_AB_module_msg(xyz, points, m, r_list, ns_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=True):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (b, n, 3) TF tensor
            points: (b, n, c) TF tensor
            m: int32 -- #points sampled in farthest point sampling
            r: list of float32 -- search radius in local region
            ns: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (b, m, 3) TF tensor
            new_points: (b, m, \sum_k{mlp[k][-1]}) TF tensor
    '''
    with tf.variable_scope(scope) as sc:

        # sampling

        new_xyz = gather_point(xyz, farthest_point_sample(m, xyz)) #(b,npoint,3)
        new_points_list = []
        print('sample ok')
        #grouping & convolution mlp
        for i in range(len(r_list)):
            print('grouping',i)
            #grouping
            r = r_list[i]
            ns = ns_list[i]
            idx, pts_cnt = query_ball_point(r, ns, xyz, new_xyz)  
            grouped_xyz = group_point(xyz, idx)  #b,m,ns[i]
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,ns,1])#(b,m,ns[i],3)
            if points is not None:
                grouped_points = group_point(points, idx)#(b,m,ns[i],c)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            print('convolutioning')
            #convolutional layers
            for j,num_out_channel in enumerate(mlp_list[i]):
                print('conv',j)
                grouped_points = basic_tf.conv2d(grouped_points, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)   #(b,m,ns[i],mlp[i][-1])

            new_points = tf.reduce_max(grouped_points, axis=[2])  #(b,m,mlp[i][-1]) 
            new_points_list.append(new_points)
        
        new_points_concat = tf.concat(new_points_list, axis=-1)#size(b,m,sum_k{mlp[k][-1])
        print('one turn')
        return new_xyz, new_points_concat


 #-----------------------------------------------------------fOR SEGMENTATION:feature propragation-----------------------------------------------------------------------------#

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True,weight_calc=True):#ok now
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (b, n1, 3) TF tensor                                                              
            xyz2: (b, n2, 3) TF tensor, sparser than xyz1                                           
            points1: (b, n1, n1) TF tensor                                                   
            points2: (b, n2, n2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (b, n1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
       
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)  
       
        if weight_calc:
            norm = tf.reduce_sum(tf.truediv(1.0,dist),axis=2,keep_dims=True)#(b,n,3)->(b,n,1)
            norm = tf.tile(norm,[1,1,3])                                #(b,n,1)->(b,n,3)
            weight = tf.truediv(tf.truediv(1.0,dist),norm)                      #(b,n,3)
        else:
            weight=get_weight(True,dist)
       
        interpolated_points = three_interpolate(points2, idx, weight)
       
        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = basic_tf .conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,n1,mlp[-1]
        return new_points1
