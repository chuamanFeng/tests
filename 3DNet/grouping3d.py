import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np

def query_ball_point(radius, nsample, xyz1, xyz2):#(ok)
    '''
    Input:
        radius: float32, ball search radius>1e-20
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    b = xyz1.get_shape()[0].value #b batches
    n = xyz1.get_shape()[1].value #n(ndataset) points   
    m = xyz2.get_shape()[1].value #m(npoint) centers     
   
    xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,3)), [1,m,1,1])#prepare for dist calculation,double [b,m,n,3] matrix
    xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,3)), [1,1,n,1])
   
    dist =tf.reshape(tf.minimum(tf.maximum(tf.reduce_sum((xyz1-xyz2)**2, -1),1e-20),radius),(b,m,n))#get the dists,store it in the tail,here the problem occurs
    
    disttemp=tf.multiply(dist,-1)+radius  
    topns,topnsid=tf.nn.top_k(disttemp,nsample)   
   
    topones=tf.tile(tf.slice(topnsid,[0,0,0],[-1,-1,1]),[1,1,nsample]) #(b,m,ns)
    topns=tf.minimum(topns,0.)     #(b,m,ns) maximum=0.
    zeroset=tf.cast(tf.equal(topns,tf.zeros_like(topns,tf.float32)),tf.int32)       #(b,m,ns)

    trans=tf.subtract(tf.ones_like(topns,tf.int32),zeroset)#(b,m,ns)
    pts_cnt=tf.reduce_sum(trans,2)   
    topnsid=tf.reshape(tf.add(tf.multiply(trans,topnsid),tf.multiply(topones,zeroset)),(b,m,nsample))   
  
    return topnsid, pts_cnt
ops.NoGradient('QueryBallPoint')   


def select_top_k_4knn(k, dist):#knn use this(ok)
    '''
    Input:
        k: int32, number of k SMALLEST elements selected
        dist: (b,m,n) float32 array, distance matrix, m query points, n dataset points
    Output:
        idx: (b,m,n) int32 array, first k in n are indices to the top k
        dist_out: (b,m,n) float32 array, first k in n are the top k
    '''  
    n=dist.get_shape()[1].value
    dist1=tf.multiply(dist,-1)
    tempd1,tempi1=tf.nn.top_k(dist1,k)
    tempd1*=-1

    tempd2,tempi2=tf.nn.top_k(dist,n-k)   

    dist_out=tf.concat([tempd1,tempd2],2)   
    idx=tf.concat([tempi1,tempi2],2)  
    
    return dist_out,idx   
           
  
def group_point(points, idx):#ok &much faster
    '''
    Input:
        points: (b,n,c) float32,   points to sample from
        idx: (b,m,ns)   int32,     indices to points
    Output:
        out: (b,m,ns,c) float32,   values sampled from points
    '''   
    b = points.get_shape()[0].value #b batches   
    n = points.get_shape()[1].value
    c = points.get_shape()[2].value
    m = idx.get_shape()[1].value #m(npoint) centers
    ns=idx.get_shape()[2].value
      
    idxn=tf.tile(tf.reshape(np.arange(n),(1,n,1,1)),[b,1,m,ns])
    idx=tf.tile(tf.reshape(idx,(b,1,m,ns)),[1,n,1,1])    #(b,n,m,ns)

    mask=tf.cast(tf.equal(idxn,idx),tf.float32)    #  0&1tensor (b,n,m,ns)
    mask=tf.tile(tf.reshape(mask,(b,n,m,ns,1)),[1,1,1,1,c])
    points=tf.tile(tf.reshape(points,(b,n,1,1,c)),[1,1,m,ns,1])    #(b,n,m,ns,c)

    out=tf.reduce_sum(tf.multiply(points,mask),1)    #(b,m,ns,c)    
   
    return out
@tf.RegisterGradient('GroupPoint')

def group_point_grad(points,idx,grad_out):#ok
    '''
input: 
    grad_out (b,m,ns,c), 
    idx (b,m,ns), 
    points(b,n,c)
output: 
    grad_points (b,n,c)

    '''
    b = grad_out.get_shape()[0].value #b batches  
    c = grad_out.get_shape()[3].value
    m = grad_out.get_shape()[1].value #m(npoint) centers
    ns= grad_out.get_shape()[2].value
    n= points.get_shape()[1].value
    nm=ns*m


    grad_out=tf.tile(tf.reshape(grad_out,(b,1,nm,c)),[1,n,1,1])
    idx=tf.tile(tf.reshape(idx,(b,1,nm)),[1,n,1])
    npts=tf.tile(tf.reshape(np.arange(n),(1,n,1)),[b,1,nm])   #(b,n,mn)

    mask=tf.cast(tf.equal(npts,idx),tf.float32)   #(b,n,mn)of 0&1
    mask=tf.tile(tf.reshape(mask,(b,n,nm,1)),[1,1,1,c])   #(b,n,mn,c)of 0&1
    grad_points=tf.multiply(tf.reduce_sum(tf.multiply(grad_out,mask),2),points) #(b,n,c)
  
    tf.reshape(grad_points,[b,n,c])
  
    return grad_points

@tf.RegisterGradient('GroupPointGrad')

def _group_point_grad(op, grad_out):#output(op):[points[b,n,c],idx[b,n,1]]
   points = op.inputs[0]
   idx = op.inputs[1]
   return group_point_grad((points, idx, grad_out), None)

def knn_point(k, xyz1, xyz2):#(ok)
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value   
    m = xyz2.get_shape()[1].value

    xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,3)), [1,m,1,1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,3)), [1,1,n,1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    
    outi, out = select_top_k_4knn(k, dist)  #here is the problem
    idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    val = tf.slice(out, [0,0,0], [-1,-1,k])
  
    val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx


if __name__=='__main__':
    knn=False
    import numpy as np
    import time
    np.random.seed(100)
    #print ('lll')
    pts = np.random.random((32,512,64)).astype('float32')
    tmp1 = np.random.random((32,512,3)).astype('float32')
    tmp2 = np.random.random((32,128,3)).astype('float32')
   
    points = tf.constant(pts)
    xyz1 = tf.constant(tmp1)
    xyz2 = tf.constant(tmp2)
    radius = 0.1 
    nsample = 64
    if knn:       
        _, idx = knn_point(nsample, xyz1, xyz2)    
        print ('knn')
        grouped_points = group_point(points, idx)
        print ('group')
    else:
        idx,_= query_ball_point(radius, nsample, xyz1, xyz2)
        print('ball')
        grouped_points = group_point(points, idx)
        grad = tf.ones_like(grouped_points)
        #points_grad = tf.gradients(grouped_points, points, grouped_points_grad)
        points_grad=group_point_grad(points,idx,grad)
    with tf.Session('') as sess:
        now = time.time() 
        for _ in range(100):
            print ('session')
            ret = sess.run(grouped_points)
        print (time.time() - now)
        