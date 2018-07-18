import numpy as np
import sys
import os
import tensorflow as tf
import PIL.Image as IM


#-------------------------------------------------------------global pack
def get_all_done(main_path,TF_savepath,img_size,file_size,brt_hue_round=1,crop_round=2,resize=True,new_size=[224,224]):

    file_folders_class=getDataFiles(main_path)  

    total_num=0
      
    for j,folder in enumerate(file_folders_class):

        label_raw=label_pre_pack(folder)
        file_num=0
        writer = tf.python_io.TFRecordWriter(TF_savepath+str(j)+str(file_num)+'.tfrecords')   

        img,label,img_num=input_prepare_pack(os.path.join(main_path,folder),img_size,label_raw,brt_hue_round,crop_round)
        
        #label=label_pre_pack(folder)    
       
        total_num+=img_num
        #curr_filesize+=img_num

        if  img_num>file_size:           
            sidx=file_size
            TFRecord_writer(writer,img[0:sidx],label[0:sidx])
            writer.close()
            file_num+=1            
            img_num-=file_size

            while img_num//file_size>0:
                writer = tf.python_io.TFRecordWriter(TF_savepath+str(j)+str(file_num)+'.tfrecords')  
                TFRecord_writer(writer,img[sidx:sidx+file_size],label[sidx:sidx+file_size])
                writer.close()

                sidx+=file_size
                img_num-=file_size
                file_num+=1
               
            writer = tf.python_io.TFRecordWriter(TF_savepath+str(file_num)+'.tfrecords')  
            TFRecord_writer(writer,img[sidx:],label[sidx:])
         
        
        else:                
            TFRecord_writer(writer,img,label)
        writer.close()    
        print(j+1,'folder(s) finished')

    print(total_num)
    writer.close()
  
#-------------------------------------------------------------local folder pack

def input_prepare_pack(filepath,img_size,label_raw,brt_hue_round=5,crop_round=6,resize=True,raw_size=[512,512],new_size=[224,224]):  #for one class

    filenames= getDataFiles(filepath)    #single class_folder
    
    img_raw,img_raw_num=image_to_array(filenames,filepath,img_size,raw_size)
    label=np.tile(np.expand_dims(label_raw,0),[img_raw_num,1]).astype('int32')
    img = change_img_dim(img_raw)
 
    c=img.shape[-1]
    if brt_hue_round>0:
        img ,label= change_brightness_hue_randomly(img,label,brt_hue_round)
    img,label = change_contrast(img,label)
    #img,label = noisy_img_randomly(img,label)

    img,label = flip_it_randomly(img,label,raw_size[0],raw_size[1])  

    img =return_to_or_shape(img,raw_size,c) 
    if crop_round>0:
        img,label = crop_randomly(img,label,raw_size,crop_round)  
    if resize:
        img,label = resize_it(img,label,raw_size,new_size)
  
    img_num=img.get_shape()[0].value

    return img,label,img_num


#------------------------------------------------------------File works
def getDataFiles(TEST_or_Train_TOPfloder):

    return os.listdir(TEST_or_Train_TOPfloder)

def label_pre_pack(folder_name):


    label_l= folder_name.split('.')
    label=[256,256,label_l[0],label_l[0],label_l[1]]
    label=np.array(label).astype('int32')
   
    label = np.reshape(label,5)
    #for labels in label_raw: 
    #    label_l=(os.path.splitext(labels))         
    #    label_l=label_l[0].split('.')
    #    listl = np.concatenate((listl, np.array(label_l[1:6])))

    #listl=np.reshape(listl.astype('int32'),(len(label_raw),5))
    #print(listl)
    return label
        
        
       
def image_to_array(filenames,filepath,img_size,raw_size): #ok

    name_num=int(len(filenames))   
    
    h=img_size[0][0]
    w=img_size[0][1]
    area=img_size[1]
 
    img = np.array([])
    
    print('transforming...')
    for name in filenames:
        image = IM.open(os.path.join(filepath,name))
        r, g, b = image.split() 
      
        r_arr = np.array(r).reshape(area)
        g_arr = np.array(g).reshape(area)
        b_arr = np.array(b).reshape(area)
       
        image_arr = np.concatenate((r_arr, g_arr, b_arr))
        img = np.concatenate((img, image_arr))
       
    img = np.reshape(img,(-1,h,w,3))   
    if h>=raw_size[0] and w>=raw_size[1]:
        orx=(w-raw_size[1])//2
        ory=(raw_size[0])//2
        img=img[:,ory:ory+512,orx:orx+512,:] 

    print('transformed')  
    '''                                             #this part have permission error
    os.chdir(savepath) 
    with open(savepath, mode='wb') as f:
        pk.dump(result, f)
    '''    
    return img.astype('float32'),int(img.shape[0])

def TFRecord_writer(writer,img,label):
     img=tf.cast(img,tf.uint8)
     label=tf.cast(label,tf.int32)
     img=(img.eval()).tobytes()  
       
     data_list={}
     for i in range(int(label.shape[1])):
         label_c=(label[:,i].eval()).tobytes()
         data_list['label_%s'%(i)]=tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_c]))
     data_list['img']=tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))

     example=tf.train.Example(features=
                              tf.train.Features(feature=data_list))
                           
     writer.write(record=example.SerializeToString())
    

#-----------------------------------------------------------------------Image works
def change_img_dim(img):

    b=img.shape[0]
    c=img.shape[-1]

    return np.reshape(img,(b,-1,c))

def change_brightness_hue_randomly(img,label,change_round=5,clip=None):
   
    if clip==None:
        clip=np.random.randint(low=0,high=50)/100
    brt=np.clip(np.random.rand(change_round),-1.*clip,clip).astype('float32')
    hue=np.clip(np.random.rand(change_round),-1.*clip,clip).astype('float32')

   
    for i in range(change_round):
        temp=tf.image.adjust_brightness(img,brt[i])
        temp=tf.image.adjust_hue(img,hue[i])
        if i==0:
            out=temp
            out_label=label
        else:
            out = tf.concat((out,temp),0)
            out_label=np.concatenate((out_label,label),0)
      
  
    return out,out_label

def change_contrast(img,label,val=None):
   
    if val==None:
        val=np.random.randint(low=0,high=30)/100   
           
    out=tf.image.adjust_contrast(img,val)
    
    return tf.concat((out,img),0),np.concatenate((label,label),0)

def flip_it_randomly(img,label,h,w):
        
    sp=np.random.randint(low=0,high=2)   
    label1=label   
    if sp==0:
        out=tf.image.flip_left_right(img)
        label1[:,0]=w-label1[:,0]        
    elif sp==1:
        out=tf.image.flip_up_down(img)
        label1[:,1]=h-label1[:,1]
    else:
        out=tf.image.flip_left_right(img)
        out=tf.image.flip_up_down(img)
        label1[:,0]=w-label1[:,0]
        label1[:,1]=h-label1[:,1]
  
    return tf.concat((out,img),0),np.concatenate((label1,label),0)

def noisy_img_randomly(img,label,c=3,weight=2):   #ok
          
    noi=2*np.random.rand(1).astype('float32')
   
    noisy_image=img+noi
    if c==3:
        noisy_image = tf.clip_by_value(noisy_image, 0.0,255.0)
    else:
        noisy_image = tf.clip_by_value(noisy_image,0.0, 1.0)
    
    return  tf.concat((noisy_image,img),0),np.concatenate((label,label),0)

def return_to_or_shape(img,img_size,c=None):
           
    b=int(img.shape[0])
    if c==None:
        c=int(img.shape[-1])

    return  tf.reshape(tf.expand_dims(img,1),(b,img_size[0],img_size[1],c))
def crop_randomly(img,label,imgsize=None,crop_round=6):


    if imgsize==None:
        h=img.shape[1]
        w=img.shape[2]
    else:
        h=imgsize[0]
        w=imgsize[1]
        
    short_edge=np.min([h,w],0)
    x0=(w-short_edge)//2
    y0=(h-short_edge)//2
    ct_img=img[:,y0:y0+short_edge,x0:x0+short_edge,:]

    crop_side=int(short_edge*0.8)
    rangex=np.random.randint(low=0,high=short_edge-crop_side,size=[crop_round])
    rangey=np.random.randint(low=0,high=short_edge-crop_side,size=[crop_round])

    for i in range(crop_round):      

        temp=ct_img[:,rangex[i]:rangex[i]+crop_side,rangey[i]:rangey[i]+crop_side,:] 
        label_t=label       
        label_t[:,0]-=rangey[i]
        label_t[:,1]-=rangex[i]
        if i==0:
            out=temp
            out_label=label_t
        else:
            out=tf.concat((out,temp),0)
            out_label=np.concatenate((out_label,label_t),0)
   
    return tf.reshape(out,(int(out.shape[0]),crop_side,crop_side,int(out.shape[-1]))),out_label
def resize_it(img,label,old_size,new_size):

    ratioy=new_size[0]/old_size[0]
    ratiox=new_size[1]/old_size[1]

    label[:,0]=label[:,0]*ratioy
    label[:,1]=label[:,1]*ratiox
       
    label[:,2]=label[:,2]* ratioy
    label[:,3]=label[:,3]* ratiox

    return tf.image.resize_images(img,new_size),label


if __name__=='__main__':

    with tf.Session() as sess:
    #img,label,num=input_prepare_pack('H:/DLtest/cool/1',[[512,512],512**2],brt_hue_round=1,resize=True,new_size=[224,224])
      get_all_done('H:/TRUEDATA/B/a','H:/TRUEDATA',[[1080,1920],1080*1920],128,brt_hue_round=2,crop_round=4,resize=True,new_size=[224,224])