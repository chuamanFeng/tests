import numpy as np
import sys
import os
import tensorflow as tf
import PIL.Image as IM


#-------------------------------------------------------------global pack
def get_all_done(main_path,TF_savepath,cloud_size,file_size):

    file_folders_class=getDataFiles(main_path)  

    total_num=0
      
    for j,folder in enumerate(file_folders_class):

        file_num=0
        writer = tf.python_io.TFRecordWriter(TF_savepath+str(j)+str(file_num)+'.tfrecords')   

        xyz,rgb,cl_num=input_prepare_pack(os.path.join(main_path,folder),cloud_size)
        label=np.tile(np.array(int(folder)),cl_num)
        #label=label_pre_pack(folder)    
       
        total_num+=cl_num
        #curr_filesize+=img_num

        if  cl_num>file_size:           
            sidx=file_size
            TFRecord_writer(writer,xyz[0:sidx],rgb[0:sidx],label[0:sidx])
            writer.close()
            file_num+=1            
            cl_num-=file_size

            while img_num//file_size>0:
                writer = tf.python_io.TFRecordWriter(TF_savepath+str(j)+str(file_num)+'.tfrecords')  
                TFRecord_writer(writer,xyz[sidx:sidx+file_size],rgb[sidx:sidx+file_size],label[sidx:sidx+file_size])
                writer.close()

                sidx+=file_size
                cl_num-=file_size
                file_num+=1
               
            writer = tf.python_io.TFRecordWriter(TF_savepath+str(file_num)+'.tfrecords')  
            TFRecord_writer(writer,xyz[sidx:],rgb[sidx:],label[sidx:])
         
        
        else:                
            TFRecord_writer(writer,xyz,rgb,label)
        writer.close()    
        print(j+1,'folder(s) finished')

    print(total_num)
    writer.close()
  
#-------------------------------------------------------------local folder pack

def input_prepare_pack(filepath,cloud_size):  #for one class

   clouds=origdata2array(filepath,cloud_size)
   clouds=rotate_it(clouds)
   clouds=noisy_it(clouds)
   clouds=jitter_it_randomly(clouds)
   b=int(clouds.shape[0])
   return clouds[:,:,:3],clouds[:,:,3:],b


#------------------------------------------------------------File works
def getDataFiles(TEST_or_Train_TOPfloder):

    return os.listdir(TEST_or_Train_TOPfloder)

def label_pre_pack(label_raw):
    listl=np.array([])
    for labels in label_raw: 
        label_l=(os.path.splitext(labels))         
        label_l=label_l[0].split('.')
        listl = np.concatenate((listl, np.array(label_l[1:6])))

    listl=np.reshape(listl.astype('int32'),(len(label_raw),5))
    print(listl)
    return listl
       
def asc2txt(filepath): 
  for filename in filepath:      
      portion = os.path.splitext(filename)
     
      if portion[1] ==".asc":
         newname = portion[0]+".txt"        
         os.rename(filename,newname) 

def origdata2array(filepath,cloud_size):    
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
   out=np.array([])
   files = os.listdir(filepath) 
  
   for filename in files:        
     if os.path.splitext(filename)[1]==(".txt"):
      
        file = open(filename)  
        list_arr = file.readlines()
        l = len(list_arr)         
        listi=[]      
        outi=[]
        for i in range(l):                   #lens in one file
            if list_arr[i].find('S')<=-1:
                list_arr[i] = list_arr[i].split()   
            if len(list_arr[i])==6:
                listi.append(list_arr[i])
              
        outi,slices=seg_randomly(np.array(listi),cloud_size)    #2 lists:[slices,n,c],(slices)   k       
            
        for k in range(slices):           
            temp=outi[k]                  
            if cnt==0:          
                out=temp
            else:              
                out=np.concatenate((out,temp),0)
            cnt+=1  
       
        #print(cnt)   
        #print(out.shape)
  
   print(cnt,'files have been pre-cooked ') 
   if cnt!=0:
     out=np.array(out).astype('float32')       
   else:
     out=None

   os.chdir(base)
   c=out.shape[-1]
  
   out=np.reshape(out,(cnt,-1,c))
    
   return out

def seg_randomly(data,size=102400): #for clouds >150000   
       
    data=np.squeeze(data)  
    nn=len(data)
    out_data=[]  
   
    n=size
    if nn<=125000:
        slices=1
       
    else:       
        slices=(nn//n)+1    
           
    for i in range(slices):                              
        np.random.shuffle(data)               
        outi=np.expand_dims(data[:n,:],0)        
       
        out_data.append(outi)
            
    return out_data,slices

def TFRecord_writer(writer,xyz,rgb,label):

     xyz=tf.cast(xyz*100000,tf.uint8)
     rgb=tf.cast(rgb,tf.uint8)
     label=tf.cast(label,tf.uint8)

     xyz=(xyz.eval()).tobytes()  
     rgb=(rgb.eval()).tobytes() 
     label=(label.eval()).tobytes()

     data_list={}
     data_list['xyz']=tf.train.Feature(bytes_list=tf.train.BytesList(value=[xyz]))
     data_list['rgb']=tf.train.Feature(bytes_list=tf.train.BytesList(value=[rgb]))
     data_list['label']=tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
     
     example=tf.train.Example(features=
                              tf.train.Features(feature=data_list))
                           
     writer.write(record=example.SerializeToString())

 #---------------------------------------------------------------------------------------#
def rotate_it(batch_data,b=None,c=None, rotation_angle=None):
    """ Rotate the point cloud along up direction with certain angle or randomly.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    if b==None:
        b=batch_data.shape[0]
    if c==None:
        c=batch_data.shape[-1]
 
    assert(c%3==0)  
    if c>3:
        batch_data1=batch_data[:,:,:3]
        batch_data2=batch_data[:,:,3:]
    else:
        batch_data1=batch_data
        batch_data2=None
    #a loop for every dataset(num b)
    rotated_data=np.zeros_like(batch_data1,float)
    for k in range(b):
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

    return np.concatenate((batch_data,rotated_data),0)

   

def jitter_it_randomly(batch_data,b=None,n=None,c=None,sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """  
    if b==None:
        b=batch_data.shape[0]
    if n==None:
        n=batch_data.shape[1]
    if c==None:
        c=batch_data.shape[-1]
  
  
    assert(clip > 0)
    jittered_data = batch_data
    #kind of like double-gates binary,add a random noise range from -cilp to clip to original point set
    jittered_data1 =np.concatenate(( np.clip(sigma * np.random.randn(b,n,3), -1*clip, clip),np.zeros_like(jittered_data[:,:,:3])),-1)
    jittered_data += jittered_data1
       
    return np.concatenate((batch_data,jittered_data ),0 )
def noisy_it(batch_data):   #ok
          
    noi=2*np.random.rand(1).astype('float32')
   
    noisy_data=np.concatenate((batch_data[:,:,:3],np.clip(batch_data[:,:,3:]+noi,0.0,255.0)),-1)
       
  
    return  np.concatenate((batch_data,noisy_data),0)



if __name__=='__main__':

    with tf.Session() as sess:
    #img,label,num=input_prepare_pack('H:/DLtest/cool/1',[[512,512],512**2],brt_hue_round=1,resize=True,new_size=[224,224])
      get_all_done('H:\D3test\A','H:\D3test',102400,128)