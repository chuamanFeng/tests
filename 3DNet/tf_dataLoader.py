import numpy as np
import sys
import os
import tensorflow as tf
import PIL.Image as IM


def TF_loader(tfr_path,cloud_size,num_epochs=None):

    if not num_epochs:
        num_epochs = None

    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(tfr_path),num_epochs)   
   
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)      
    features = tf.parse_single_example(serialized_example,
                                       features=
                                       {
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'rgb' : tf.FixedLenFeature([], tf.string),
                                           'xyz': tf.FixedLenFeature([], tf.string),
                                        }
                                       )

    label = tf.decode_raw(features['label'], tf.uint8)  
    rgb = tf.decode_raw(features['rgb'], tf.uint8)      
    xyz = tf.decode_raw(features['xyz'], tf.uint8)   

    rgb = tf.reshape(rgb, [cloud_size[0],cloud_size[1],-1])  
    rgb = tf.cast(tf.transpose(rgb,[2,0,1]),tf.float32) /255.
    xyz = tf.reshape(xyz, [cloud_size[0],cloud_size[1],-1])  
    xyz =tf.cast( tf.transpose(xyz,[2,0,1]),tf.float32)/100000


    return tf.concat((xyz,rgb),-1),label

#def get_file_shape(image):
#   return int(image.shape[0])
def TF_loader_multi(root_path,cloud_size,num_epochs=None):
     
 
    list1=[]
    list2=[]
    filenames=os.listdir(root_path)
    print(filenames)
    for i,name in enumerate(filenames):
            path=os.path.join(root_path,name)
            print(path)
            cloud,label=TF_loader(path,cloud_size,num_epochs=None)
            list1.append(cloud)
            list2.append(label)
  
    return list1,list2



def shuffle_it(cloud,label,b=None,n=None):
   
    if b==None:
        b=int(cloud.shape[0])

    cloud=shuffle_within(cloud,n)

    idx=np.arange(b)
    np.random.shuffle(idx)
    
   
    return cloud[idx],label[idx]

def shuffle_within(cloud,n=None):

    if n==None:
        n=int(cloud._shape[1])

    cloud=np.transpose(cloud,[1,0,2])
    idx=np.arange(n)
    np.random.shuffle(idx)

    cloud=cloud[idx]
    return np.transpose(cloud,[1,0,2])

def data_loader(list1,list2):

    for i in range(len(list1)):
        cloud=list1[i]
        label=list2[i]
        if int(label.shape[0])==1:
            num=int(cloud.shape[0])
            label=np.tile(label,num)

        if i==0:
            out_cloud=cloud
            out_label=label
        else:
            out_cloud=np.concatenate((out_cloud,cloud),0)           
            out_label=np.concatenate((out_label,label),0)
   
    out_cloud,out_label=shuffle_it(out_cloud,out_label)
   
    return out_cloud,out_label

 
def full_path_maker(dir_list,file_list):

    list_out=[]
    for i,dir in enumerate(dir_list):
        list_temp=[]
        names=file_list[i]
        for j,name in enumerate(names):
            list_temp.append(os.path.join(dir,name))

        list_out.append(list_temp)

    return list_out



if __name__=="__main__":

    #with tf.Session() as sess:
    #    l1,l2=data_loader('H:/DLtest/C',[224,224,3],None)
    #    print(l1)
    list1,list2=TF_loader_multi('H:\D3test\B\pack0',[102400,6],num_epochs=None)
    sv = tf.train.Supervisor()  
    with sv.managed_session() as sess: 
        list1,list2=sess.run([list1,list2])
        
        print(list2)

            