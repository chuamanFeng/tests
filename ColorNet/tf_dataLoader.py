import numpy as np
import sys
import os
import tensorflow as tf
import PIL.Image as IM


def TF_loader(tfr_path,img_size,num_epochs=None):

    if not num_epochs:
        num_epochs = None

    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(tfr_path),num_epochs)   
   
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)      
    features = tf.parse_single_example(serialized_example,
                                       features=
                                       {                                      
                                           'img' : tf.FixedLenFeature([], tf.string),
                                        }
                                       )

    image = tf.decode_raw(features['img'], tf.uint8)         
    image = tf.reshape(image, [img_size[0],img_size[1],img_size[2],-1])  
    image = tf.transpose(image,[3,0,1,2]) 
  
 

    return image
#def get_file_shape(image):
#   return int(image.shape[0])
def TF_loader_multi(path_rgb,path_srgb,img_size,num_epochs=None):     
 
    list1=[]
    list2=[]
    filenames=os.listdir(path_rgb)
    filenames2=os.listdir(path_srgb)
    boo,mini=check_pair_len(filenames,filenames2)

    if boo==False:        
        filenames=filenames[:mini]
        filenames2=filenames2[:mini]

    for i in range(mini):
            path1=os.path.join(path_rgb,filenames[i])
            path2=os.path.join(path_srgb,filenames2[i])

            rgb=TF_loader(path1,img_size,num_epochs=None)
            srgb=TF_loader(path2,img_size,num_epochs=None)

            list1.append(rgb)
            list2.append(srgb)
   
  
    return list1,list2

    
def shuffle_it(rgb,srgb,b=None):
   
    if b==None:
        b=int(rgb.shape[0])
    
    idx=np.arange(b)
    np.random.shuffle(idx)
    
   
    return rgb[idx],srgb[idx]


def data_loader(list1,list2):

    for i in range(len(list1)):
        rgb=list1[i]
        srgb=list2[i]
        if check_dims(rgb,srgb):
            if i==0:
                out_rgb=rgb
                out_srgb=srgb
            else:
                out_rgb=np.concatenate((out_rgb,rgb),0)
                out_srgb=np.concatenate((out_srgb,srgb),0)
   
    out_rgb,out_srgb=shuffle_it(out_rgb,out_srgb)
    out_rgb=out_rgb/255.0
    out_srgb=out_srgb/255.0

   
    return out_rgb,out_srgb

 
def full_path_maker(dir_list,file_list):

    list_out=[]
    for i,dir in enumerate(dir_list):
        list_temp=[]
        names=file_list[i]
        for j,name in enumerate(names):
            list_temp.append(os.path.join(dir,name))

        list_out.append(list_temp)

    return list_out

def check_pair_len(list1,list2):

    max=np.max([len(list1),len(list2)])
    min=np.min([len(list1),len(list2)])
    
    return bool(max==min),min

def check_dims(rgb,srgb):
    
    return bool(rgb.shape==srgb.shape)     

def origin_list(image_size,crop_size):

    H=image_size[0]
    W=image_size[1]
    h=crop_size[0]
    w=crop_size[1]
    if H<h or W<w:
        print('crop_size too big')
        return None

    
    n=np.min([int(H/h),int(W/w)],0)
    plusx=(W-n*w)//2
    plusy=(H-n*h)//2
    arr=np.arange(n)
    list_x=np.expand_dims((arr)*w+plusx,-1)
    list_y=np.expand_dims((arr)*h+plusy,-1)


    out=np.concatenate((list_x,list_y),-1)
    print(out)
    return out


if __name__=="__main__":

    with tf.Session() as sess:
        l1,l2=data_loader('H:/DLtest/C',[224,224,3],None)
        print(l1)
    #list1,list2=TF_loader_multi('H:/DLtest/C',[224,224,3],num_epochs=None)
    #sv = tf.train.Supervisor()  
    #with sv.managed_session() as sess: 
    #    list1,list2=sess.run([list1,list2])
            