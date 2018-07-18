import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np
import time

def gather_point(inp,idx):     #ok     faster in 10.16                                     
    '''
input:
    b * n * c        float32
    b * ns           int32 (idx)   
returns:
    b * ns * c list  float32
    '''      
    b = inp.get_shape()[0].value #b batches  
    n = inp.get_shape()[1].value
    c = inp.get_shape()[2].value
    ns= idx.get_shape()[1].value

    idxn=tf.tile(tf.reshape(np.arange(n),(1,n,1)),[b,1,ns])
    idx=tf.tile(tf.reshape(idx,(b,1,ns)),[1,n,1])    #(b,n,ns)

    mask=tf.cast(tf.equal(idxn,idx),tf.float32)    #  0&1tensor (b,n,ns)
    mask=tf.tile(tf.reshape(mask,(b,n,ns,1)),[1,1,1,c])
    inp=tf.tile(tf.reshape(inp,(b,n,1,c)),[1,1,ns,1])

    outgather=tf.reduce_sum(tf.multiply(inp,mask),1)

    tf.reshape(outgather,(b,ns,c))   
    print('gather_ok')

    return outgather
@tf.RegisterGradient('GatherPoint')


def gather_point_grad(points,idx,grad_out): #ok
    '''
input:
    b * n * c  float32
    b * (m * ns)    int32   
    b * (m * ns) * c 
returns:
    b * n *c   float32
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
    print('22')
    return grad_points

@tf.RegisterGradient('GatherPointGrad')
                                                                   
def _gather_point_grad(op,out_g):                                       

    inp=op.inputs[0]
    idx=op.inputs[1]
   

    return gather_point_grad(inp,idx,out_g)
                                                                                 
def farthest_point_sample_small(npoint,inp):   #ok in 10.17,speed up 3 times
   
    b=inp.get_shape()[0].value
    n=inp.get_shape()[1].value
    c=inp.get_shape()[2].value
   
    idxn=tf.tile(tf.reshape(np.arange(1,n+1),(1,n)),[b,1])    #(b,n)     

    startid=np.random.random_integers(1,high=n,size=[b,1])
    out=startid                          #here we have (b,1)already,so we only need to search npoint-1 times!fucking stupid(10.18)
    startid=tf.tile(startid,[1,n]) #(b,n)
  

    mask=tf.tile(tf.reshape(tf.cast(tf.equal(startid,idxn),tf.float32),(b,n,1)),[1,1,c+1])  #(b,n,c+1) 0&1tensor  
    new_start=tf.concat([inp,tf.reshape(tf.cast(idxn,tf.float32),(b,n,1))],-1)    #(b,n,c+1) 
    #make a dict to store pts in each round
    new_ones={}
    new_ones['loop_pts']=new_start
     
    for i in range(npoint-1):
        if i<n-1:   
            new_starti=new_ones['loop_pts']     
            
            if i==0:
                startpoints=tf.tile(tf.reduce_sum(tf.multiply(mask, new_starti),1,True),[1,n-i,1])   #(b,n,c+1)               
            else:
                startpoints=tf.tile(tf.slice(new_starti,[0,0,0],[-1,1,-1]),[1,n-i,1])   #(b,n-i,c+1) top one


            startgroup=new_starti                        #(b,n-i,c+1) with top-one                             
            temp=(startpoints-startgroup)**2  #(b,n-i,c+1)
            dists=tf.reduce_sum(tf.slice(temp,[0,0,0],[-1,-1,c]),-1) #(b,n-i)
           
            _,idtpc=tf.nn.top_k(dists,n-i-1)  #(b,n-i)ids,sorted,the last one is it self,we expel it.
            idtpc+=1   #plus one make it available
           
            idtopc=tf.tile(tf.slice(idtpc,[0,0],[-1,1]),[1,n-i])  #toppoint id ,sorted(b,1),expand to(b,n-i)    
            idxni=tf.slice(idxn,[0,0],[-1,n-i])  #(b,n-i),preparing, numb from1~n-i,     int32
            
            #now searching its origin id (size:(b,1)) from new_start
            maski=tf.tile(tf.reshape(tf.cast(tf.equal(idtopc,idxni),tf.float32),(b,n-i,1)),[1,1,c+1]) #(b,n-i,c+1) 
            outi=tf.slice(tf.reduce_sum(tf.multiply(maski,new_starti),1),[0,c],[-1,-1])
           
            #now prepare the next pts_group,with chosen ones in this round            
            idtpc=tf.tile(tf.reshape(idtpc,(b,n-i-1,1)),[1,1,n-i])       #(b,n-i-1,n-i)
            idxni=tf.tile(tf.reshape(idxni,(b,1,n-i)),[1,n-i-1,1])            
            maski2=tf.tile(tf.cast(tf.reshape(tf.equal(idxni,idtpc),(b,n-i-1,n-i,1)),tf.float32),[1,1,1,c+1] )     #(b,n-i-1,n-i)           
            new_ones['loop_pts']=tf.reduce_sum(tf.multiply(maski2,tf.tile(tf.reshape(new_starti,(b,1,n-i,c+1)),[1,n-i-1,1,1])),2)#(b,n-1-i,n-i,c+1)->(b,n-i-1,c+1)
            
            out=tf.concat([out,outi],-1)#(float32)
        else:            
             break
      
    out=tf.maximum(tf.subtract(tf.cast(out,tf.int32),1),0)    #why in second round it turns to 129???
  
    return tf.reshape(out,(b,npoint))
ops.NoGradient('FarthestPointSampleSmall') 
                                                                                   
def farthest_point_sample(npoint,inp):   #ok in 10.17,speed up 3 times        
    n=inp.get_shape()[1].value
    if n>=npoint:       
        if npoint>=512:
            b=inp.get_shape()[0].value
            n=inp.get_shape()[1].value
            c=inp.get_shape()[2].value
            idxn=tf.tile(tf.reshape(np.arange(1,n+1),(1,n)),[b,1])    #(b,n) 
            new_start=tf.concat([inp,tf.reshape(tf.cast(idxn,tf.float32),(b,n,1))],-1)    #(b,n,c+1) ,point set with idx in last channel
            #n0=n //npoint  #int :rate of orig&samp 2,get the min side
            
            if npoint>=5120:
                split_int=256
            elif npoint>=2048:
                split_int=64
            elif npoint>=1024:
                split_int=16
            else:
                split_int=4

            n1=split_int
            n2=n//split_int 
            npoint_mini=npoint//split_int
            if n%split_int!=0:   
                
                pre_sam=n1*n2 
                new_start=pre_sample_randomly(new_start,pre_sam)
        
          
            new_start=tf.reshape(new_start,(b,n1,n2,c+1))
    
            #prepare start idxs in each n2-size group
            startid=tf.tile(tf.expand_dims(np.random.random_integers(1,high=n2,size=[n1,1]),0),[b,1,1])#(b,n1,1)
            idxn_mini=tf.tile(tf.reshape(np.arange(1,n2+1),(1,1,n2)),[b,n1,1])                       
            startid=tf.tile(startid,[1,1,n2]) #(b,n)
  
            mask=tf.tile(tf.reshape(tf.cast(tf.equal(startid,idxn_mini),tf.float32),(b,n1,n2,1)),[1,1,1,c+1])  #(b,n1,n2,c+1) 0&1tensor  
            startpoint=tf.reduce_sum(tf.multiply(mask,new_start),2) #(b,n1,n2,c+1)->(b,n1,c+1)
            out=tf.slice(startpoint,[0,0,c],[-1,-1,-1])             #(b,n1,c+1)->(b,n1,1)
            #make a dict to store pts in each round
            new_ones={}
            new_ones['loop_pts']=new_start #(b,n1,n2,c+1)

            idxn0=tf.tile(tf.reshape(np.arange(1,n2+1),(1,1,n2)),[b,n1,1])     #(b,n1,n2)
            for i in range(npoint_mini-1):
                if i<n2-1:   
                    new_starti=new_ones['loop_pts']     
            
                    if i==0:
                        startpoints=tf.tile(tf.reshape(startpoint,(b,n1,1,c+1)),[1,1,n2-i,1])   #(b,n1,n2-i,c+1)               
                    else:
                        startpoints=tf.tile(tf.slice(new_starti,[0,0,0,0],[-1,-1,1,-1]),[1,1,n2-i,1])   #(b,n1,n2-i,c+1) top one


                    startgroup=new_starti                        #(b,n1,n2-i,c+1) with top-one      799&800?                       
                    temp=(startpoints-startgroup)**2  #(b,n1,n2-i,c+1)
                    dists=tf.reduce_sum(tf.slice(temp,[0,0,0,0],[-1,-1,-1,c]),-1) #(b,n1,n2-i)
           
                    _,idtpc=tf.nn.top_k(dists,n2-i-1)  #(b,n1,n2-i-1)ids,sorted,the last one is it self,we expel it.
                    idtpc+=1   #plus one make it available
           
                    idtopc=tf.tile(tf.slice(idtpc,[0,0,0],[-1,-1,1]),[1,1,n2-i])  #toppoint id ,sorted(b,1),expand to(b,n1,n2-i)   
                    idxni=tf.slice(idxn0,[0,0,0],[-1,-1,n2-i])     #(b,n1,n2-i)
                    #idxni=tf.tile(tf.reshape(np.arange(1,n2-i+1),(1,1,n2-i)),[b,n1,1])  #(b,n-i),preparing, numb from1~n-i,     int32
            
                    #now searching its origin id (size:(b,1)) from new_start
                    maski=tf.tile(tf.reshape(tf.cast(tf.equal(idtopc,idxni),tf.float32),(b,n1,n2-i,1)),[1,1,1,c+1]) #(b,n1,n2-i,c+1) 
                    outi=tf.slice(tf.reduce_sum(tf.multiply(maski,new_starti),2),[0,0,c],[-1,-1,-1])  #(b,n1,n2-i,1)
           
                    #now prepare the next pts_group,with chosen ones in this round            
                    idtpc=tf.tile(tf.expand_dims(idtpc,3),[1,1,1,n2-i])       #(b,n1,n2-i-1,n2-i)
                    idxni=tf.tile(tf.expand_dims(idxni,2),[1,1,n2-i-1,1])            
                    maski2=tf.tile(tf.cast(tf.expand_dims(tf.equal(idxni,idtpc),4),tf.float32),[1,1,1,1,c+1] )     #(b,n1,n2-i-1,n2-i,c+1)         
                    new_ones['loop_pts']=tf.reduce_sum(tf.multiply(maski2,tf.tile(tf.expand_dims(new_starti,2),[1,1,n2-i-1,1,1])),3)#(b,n1,n2-1-i,n2-i,c+1)->(b,n1,n2-i-1,c+1)
            
                    out=tf.concat([out,outi],-1)#(float32) (b,n1*npoint_mini)
                else:            
                        break
        
            out=tf.reshape(tf.maximum(tf.subtract(tf.cast(out,tf.int32),1),0),(b,npoint))   #why in second round it turns to 129???
        else:
            out=farthest_point_sample_small(npoint,inp)  
    else:
        print('sample number larger than the set size')
        out=None
             
    return out
ops.NoGradient('FarthestPointSample')

def pre_sample_randomly(data,n):     
     B=data.get_shape()[0].value
     nn=data.get_shape()[1].value
     c=data.get_shape()[2].value
     #data+=1e-20             #to avoid some (0.,0.,0.)point    
     
     tempidx=np.arange(nn)    
     np.random.shuffle(tempidx)
     #print(tempidx)
     chosen=tf.slice(tempidx,[0],[n])
          
     chosenid=tf.cast(tf.tile(tf.reshape(chosen,[1,n,1]),[B,1,nn]) ,tf.float32) #(B,nn,c)
     tempidx=tf.cast(tf.tile(tf.reshape(tempidx,[1,1,nn]),[B,n,1]) ,tf.float32)
     

     mask=tf.cast(tf.equal(chosenid,tempidx),tf.float32)
     mask=tf.tile(tf.expand_dims(mask,3),[1,1,1,c])     #(B,nn,n,c)
    # mask_res=tf.ones_like(mask)-mask

     out=tf.reduce_sum(tf.multiply(mask,tf.tile(tf.expand_dims(data,1),[1,n,1,1])),2)     #(B,n,c)
     #res=tf.reduce_sum(tf.multiply(mask_res,tf.tile(tf.expand_dims(data,1),[1,n,1,1])),2)
     #out=tf.reshape(out,(B,n,c))-1e-20
   
     return out
ops.NoGradient('PreSample')

if __name__=='__main__':    
    import numpy as np    
    np.random.seed(100)
    #print ('lll')
    pts = np.random.random((32,102488,3)).astype('float32')
    grads = np.random.random((32,32,32,3)).astype('float32')  #b=m=ns=32
    idx= np.random.random_integers(0,512,(32,32,32))

    #input1=tf.constant(pts)
    #input2=tf.constant(grads)
    #input3=tf.constant(idx)

    #gather_point_grad(input1,input3,input2)
    input=tf.constant(pts)
    now = time.time() 
    #idx=farthest_point_sample(64,input)
    idx=farthest_point_sample(51200,input)
    print (time.time() - now)
    output= gather_point(input,idx)
    