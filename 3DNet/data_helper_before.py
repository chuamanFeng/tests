import os
import sys
import numpy as np
import h5py
import tensorflow as tf
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def asc2txt(filepath): 
  for filename in filepath:      
      portion = os.path.splitext(filename)
     
      if portion[1] ==".asc":
         newname = portion[0]+".txt"        
         os.rename(filename,newname) 
      
#---------------------------------------for single-dim label set---------------------#
def origdata2array(filepath):    
   '''
   handling both labels and dataset.(generate single-dim labels'array)
outputs:
   label array&data array(without segment) 
   '''
   base=os.getcwd()
   os.chdir(filepath)     
   files = os.listdir(filepath) 
   asc2txt(files)     
   cnt=0
  
   files = os.listdir(filepath) 
   for filename in files:        
     if os.path.splitext(filename)[1]==(".txt"):
       
        file = open(filename)  
        list_arr = file.readlines()
        l = len(list_arr)         
        listi=[]
        labeli=[]
        outi=[]
        for i in range(l):                   #lens in one file
            if list_arr[i].find('S')<=-1:
                list_arr[i] = list_arr[i].split()   
            if len(list_arr[i])==6:
                listi.append(list_arr[i])  
          
        outi,labeli,slices=seg_randomly(np.array(listi),filename)    #2 lists:[slices,n,c],(slices)   k 
        outi=np.expand_dims(outi,1)      
        for k in range(slices):           
            temp=outi[k]          
            templ=labeli[k]
            if cnt==0:                
                out_label=templ
                out=temp
            else:
                out_label=np.hstack((out_label,templ))
                out=np.hstack((out,temp))
            cnt+=1  
       
        print(cnt)   
        print(out.shape)
  
   print(cnt,'files have been pre-cooked ')

   if cnt!=0:
     out=np.array(out).astype('float32')      
     out_label=np.array(out_label).astype('str')
   else:
       out=out_label=None

   os.chdir(base)
   c=out.shape[-1]
  
   out=np.reshape(out,(cnt,-1,c))
   out_label=np.reshape(out_label,(cnt))

   print(out.shape)
   return out,out_label 

def seg_randomly(data,label,size=102400): #for clouds >150000   
       
    data=np.squeeze(data)
    label=np.squeeze(label)
    nn=len(data)
    out_data=[]  
    out_label=[] 
    n=size
    if nn<=125000:
        slices=1
       
    else:       
        slices=(nn//n)+1    
           
    for i in range(slices):
                              
        np.random.shuffle(data)               
        outi=data[:n,:]
        print(outi.shape)
        out_data.append(outi)
        out_label.append(label)      
     
    return out_data,out_label,slices 

def seg_into_batches(data,label,batch_size=32,auto_arange=True):
    '''
inputs
    data:B,..,C numpy array
    label:B,...numpy array
    batch_size:int(default=32)
outputs:
    data_out:b1*batch_size,...,c datasets
    label_out:b1*batch_size,.. labelsets
    '''
    b=batch_size
    B=data.shape[0]  
    print(B)
    c=data.shape[2]
    b1=B//b 
    print(b1)
    w=b1*b
   
    data1=data[:w,:,:]
    label1=label[:w]
    data_out=np.reshape(np.expand_dims(data1,0),(b1,b,-1,c))
    label_out=np.reshape(np.expand_dims(label1,0),(b1,b))
       
   
    if B%b!=0:
        if auto_arange:
            b1r=B-w
            resdata=data[w:,:,:]
            resl=label[w:]
            if b1r<(b/2):
                b1r*=2
                resdata=np.concatenate((resdata,resdata),0)
                resl=np.concatenate((resl,resl),0)

            need=b-b1r            
            temp=data[:w,:,:]
            templ=label[:w]
            idx=np.arange(w)
            np.random.shuffle(idx)
            temp=temp[idx]
            templ=templ[idx]

            chosen=temp[:need,:,:]
            chosenl=templ[:need]
            data_out=np.concatenate((data_out,np.expand_dims(np.concatenate((chosen,resdata),0),0)),0)    #b1+1
            label_out=np.concatenate((label_out,np.expand_dims(np.concatenate((chosenl,resl),0),0)),0)
           
            

    out_data=np.array(data_out).astype('float32')
    out_label=np.array(label_out).astype('str')
    print(out_data.shape,out_label.shape)    
    return out_data,out_label  

def prepare_segmentation_label_set(data,label,normed=True):

    label=np.expand_dims(label,-1)
   
    b1=data.shape[0]   #(b1,B,n,c)or(B,n,c)
    B=data.shape[-3]
    c=data.shape[-1]         
    W=b1
    if B!=b1:     #4 dims
        W*=B      
        labels=np.reshape(labels,(W,1))

    if normed:
        N=data.shape[-2]
        label_out=np.tile(label,[1,N])      #(W,N)
    if normed==False:
        data=np.reshape(data,(W,-1,c))
        label=np.expand_dims(label,1)   #(W,1,1)
        for i in range(W):
            ni=data[i].shapep[0]     #(ni,c)  
            label[i]=np.tile(label[i],[1,ni])   #(1,1)->(1,ni)
            if i==0:
                label_out=label[i]
            else:
                label_out=np.hstack((label_out,label[i]))
        label_out=np.reshape(label_out,(W,-1))
     
    if W!=b1:
        label_out=np.reshape(label_out,(b1,B,-1))   
    
    return label_out
   

def shuffle_it(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    b1=data.shape[0]   #(b1,B,n,c)or(B,n,c)
    B=data.shape[-3]
    c=data.shape[-1]         
    W=b1
    if B!=b1:     #4 dims
        W*=B 
        data=np.reshape(data,(W,-1,c))  
        labels=np.reshape(labels,(W))
    idx = np.arange(W)
    np.random.shuffle(idx)
    data=data[idx, ...]
    labels=labels[idx]

    if W!=b1:
        data=np.reshape(data,(b1,B,-1,c))   
        labels=np.reshape(labels,(b1,B))

    return data, labels, idx  


def rotate_it(batch_data, rotation_angle=None):
    """ Rotate the point cloud along up direction with certain angle or randomly.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    b1=batch_data.shape[0]
    B=batch_data.shape[-3]
    c=batch_data.shape[-1]
    W=b1
    if B!=b1:     #4 dims
        W*=B 
        batch_data=np.reshape(batch_data,(W,-1,c))
           
    assert(c%3==0)  
    if c>3:
        batch_data1=batch_data[:,:,:3]
        batch_data2=batch_data[:,:,3:]
    else:
        batch_data1=batch_data
        batch_data2=None
    #a loop for every dataset(num b)
    rotated_data=np.zeros_like(batch_data1,float)
    for k in range(W):
        if rotation_angle==None:
            rotation_angle = np.random.uniform() * 2 * np.pi

        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data1[k, ...]          #this sentence support vary shapes in each batch,very cool
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)    
    
    rotated_data=np.concatenate((rotated_data,batch_data2),-1)
    if W!=b1:
        rotated_data=np.reshape(rotated_data,(b1,B,-1,c))

    return rotated_data

   

def jitter_it_randomly(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """  
    b1=batch_data.shape[0]
    B=batch_data.shape[-3]
    N=batch_data.shape[-2]
    c=batch_data.shape[-1]
    W=b1
    if B!=b1:     #4 dims
        W*=B 
        batch_data=np.reshape(batch_data,(W,-1,c))
           
    assert(clip > 0)

    #kind of like double-gates binary,add a random noise range from -cilp to clip to original point set
    jittered_data = np.clip(sigma * np.random.randn(W,N,c), -1*clip, clip) 
    jittered_data += batch_data

    if W!=b1:
        jittered_data=np.reshape(jittered_data,(b1,B,N,c))
    return jittered_data

def loadDataFile(filepath,batch_size,seg=False):

    lablenames =  os.listdir(filepath)    #the big one,get a list of folder names
    cnt=0
    for name in lablenames:
         child = os.path.join('%s%s' % (filepath, name))
         data,label=origdata2array(child)
        
         if seg:
             label=prepare_segmentation_label_set(data,label)
         if cnt==0:
             out_data=data
             out_label=label
         if cnt>0:
             out_data=np.hstack((out_data,data))
             out_label=np.hstack((out_label,label))
         cnt+=1
              
    out_data,out_label=seg_into_batches(out_data,out_label,batch_size)
    return data,label
#def getDataFiles(list_filename):
#    return [line.rstrip() for line in open(list_filename)]

#def load_h5(h5_filename):
#    f = h5py.File(h5_filename)
#    data = f['data'][:]
#    label = f['label'][:]
#    return (data, label)

#def loadDataFile(filename):
#    return load_h5(filename)

#def load_h5_data_label_seg(h5_filename):
#    f = h5py.File(h5_filename)
#    data = f['data'][:]
#    label = f['label'][:]
#    seg = f['pid'][:]
#    return (data, label, seg)


#def loadDataFile_with_seg(filename):
#    return load_h5_data_label_seg(filename)

if __name__=='__main__':
    
     #data = np.random.random((64,150000,6)).astype('float32')
     #labels = np.arange(64)    
     now=time.time()
     data,label=origdata2array('C:\Aaa')     #30sec for 27 clouds' output
     data1,label1=seg_into_batches(data,label,16)
     data2,label2,_=shuffle_it(data1,label1)
     data3=rotate_it(data2)
     data4=jitter_it_randomly(data3)
     print(data4.shape)    #(16,102400,6)                 #
     #outdata,outlable=seg_bigones_randomly(data,labels)    #7.7sec
     #print(outlable)
     print(time.time()-now)