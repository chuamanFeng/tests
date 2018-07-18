import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np
def three_nn(xyz1, xyz2):#ok
    '''
    Input:
        xyz1: (b,n,3) float32, unknown points (32,129,3)
        xyz2: (b,m,3) float32, known points   (32,1,3)
    Output:
        dist: (b,n,3) float32, distances to known points
        idx: (b,n,3) int32, indices to known points
    '''
    b=xyz1.get_shape()[0].value #(32)
    n=xyz1.get_shape()[1].value #129
    m=xyz2.get_shape()[1].value #1  

    xyz1=tf.tile(tf.reshape(xyz1,(b,n,1,3)),[1,1,m,1])
    xyz2=tf.tile(tf.reshape(xyz2,(b,1,m,3)),[1,n,1,1]) #b,n,m,3
     
    dists=tf.reshape(tf.reduce_sum((xyz1-xyz2)**2,3),(b,n,m)) #(32,129,1)   

    if m==1:
        idx=tf.zeros([b,n,3],tf.int32)
        dist=tf.tile(dists,[1,1,3])
    elif m>=3:
        dist,idx=tf.nn.top_k(tf.multiply(dists,-1),3)                #(32,1,3)???

    return tf.reshape(dist,(b,n,3)),tf.reshape(idx,(b,n,3))
   

def get_weight(constant=True,dist=None):#ok
    if dist!=None:
        b=dist.get_shape()[0].value
        n=dist.get_shape()[1].value
    else:
        weight=tf.truediv(tf.ones_like(dist),3.0)

    if constant==True:
        weight=tf.truediv(tf.ones_like(dist),3.0)
    else:        
        distsum=tf.tile(tf.reduce_sum(dist,-1,True),[1,1,3]) #(b,n,1)->(b,n,3)        
        weight=tf.reshape(tf.div(dist,distsum),(b,n,3))

    return weight

def three_interpolate(points, idx, weight):#ok and faster now
    '''
    Input:
        points: (b,m,c) float32, known points
        idx: (b,n,3) int32, indices to known points
        weight: (b,n,3) float32, weights on known points
    Output:
        out: (b,n,c) float32, interpolated point values
    '''
    print('start')
    b=points.get_shape()[0].value
    n=idx.get_shape()[1].value
    m=points.get_shape()[1].value
    c=points.get_shape()[2].value

    idxn=tf.tile(tf.reshape(np.arange(m),(1,1,1,m)),[b,n,3,1])
    idx=tf.tile(tf.reshape(idx,(b,n,3,1)),[1,1,1,m])    #(b,n,3,m)

    mask=tf.cast(tf.equal(idxn,idx),tf.float32)    #  0&1tensor (b,n,3,m)
    mask=tf.tile(tf.reshape(mask,(b,n,3,m,1)),[1,1,1,1,c])
    points=tf.tile(tf.reshape(points,(b,1,1,m,c)),[1,n,3,1,1])    #(b,n,3,m,c)

    gather=tf.reduce_sum(tf.multiply(points,mask),3)    #(b,n,3,c)
  
    tf.reshape(gather,(b,n,3,c))  

    weight=tf.tile(tf.reshape(weight,(b,n,3,1)),[1,1,1,c])
    out=tf.reshape(tf.reduce_sum(tf.multiply(gather,weight),2),(b,n,c))
    print('okk')
    return out
  
def three_interpolate_grad(grad_out,idx,weight,points):  #ok
    '''
input: 
    grad_out (b,n,c), 
    idx (b,n,3), 
    weight (b,n,3)  (every out_grad has 3 weights to 3 original points,which append to 3 idxes)
output:
    grad_points (b,m,c)
    ''' 
    b=points.get_shape()[0].value
    n=idx.get_shape()[1].value  
    m=points.get_shape()[1].value
    c=points.get_shape()[2].value

   
    grad_out=tf.tile(tf.reshape(grad_out,(b,n,1,c)),[1,1,3,1])           #(b,n,3,c)
    weight=tf.tile(tf.reshape(weight,(b,n,3,1)),[1,1,1,c])
    grad_fi=tf.reshape(tf.multiply(grad_out,weight),(b,3*n,c))    #(b,3*n,c)
       

    idx=tf.tile(tf.reshape(idx,(b,1,n*3)),[1,m,1])
    idxn=tf.tile(tf.reshape(np.arange(m),(1,m,1)),[b,1,3*n])   
    mask=tf.tile(tf.reshape(tf.cast(tf.equal(idx,idxn),tf.float32),(b,m,3*n,1)),[1,1,1,c])               #(b,m,3*n,c) 0&1tensor 

    grad_mask=tf.multiply(tf.tile(tf.reshape(grad_fi,(b,1,3*n,c)),[1,m,1,1]),mask)    #(b,m,3*n,c)
    grad_points=tf.multiply(points,tf.reduce_sum(grad_mask,2))    #(b,m,c)

    #print('good') 

    return tf.reshape(grad_points,(b,m,c))
 
   

if __name__=='__main__':
    import numpy as np
    import time
    np.random.seed(100)
    pts = np.random.random((32,128,64)).astype('float32')
    tmp1 = np.random.random((32,512,3)).astype('float32')
    tmp2 = np.random.random((32,128,3)).astype('float32')  #m<n
    with tf.device('/cpu:0'):
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        dist, idx = three_nn(xyz1, xyz2)
        #weight = tf.ones_like(dist)/3.0  #新建一个与给定的tensor（dist）类型大小一致的tensor,其所有元素为1(此处/3，代表权值各占1/3)
        weight=get_weight(False,dist)
        interpolated_points = three_interpolate(points, idx, weight)
        #grad_out=tf.ones([32,512,64],tf.float32)
        #grad_points=three_interpolate_grad(grad_out,idx,weight,points)
    with tf.Session('') as sess:
        now = time.time() 
        for _ in range(100):
            ret = sess.run(interpolated_points)
            print('ok')
        #print time.time() - now
        #print ret.shape, ret.dtype
        #print ret
