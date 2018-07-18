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

        file_num=0
        writer = tf.python_io.TFRecordWriter(TF_savepath+str(j)+str(file_num)+'.tfrecords')   

        img,img_num=input_prepare_pack(os.path.join(main_path,folder),img_size,brt_hue_round,crop_round)
        
        #label=label_pre_pack(folder)    
        label_l=(os.path.splitext(folder))         
        label_l=label_l[0].split('.')
        label=np.tile(np.array((int(label_l[0]))),img_num)
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

def input_prepare_pack(filepath,img_size,brt_hue_round=5,crop_round=6,resize=True,raw_size=[512,512],new_size=[224,224]):  #for one class

    filenames= getDataFiles(filepath)    #single class_folder
    img_raw=image_to_array(filenames,filepath,img_size,raw_size)

    img = change_img_dim(img_raw)
   
    c=img.shape[-1]
    if brt_hue_round>0:
        img = change_brightness_hue_randomly(img,brt_hue_round)
    img = change_contrast(img)
    #img = noisy_img_randomly(img)
    img = flip_it_randomly(img)  

    img =return_to_or_shape(img,raw_size,c)
    if crop_round>0:
        img = crop_randomly(img,raw_size,crop_round)
    if resize:
        img = resize_it(img,new_size)
    
    img_num=img.get_shape()[0].value
    return img,img_num
def label_pre_pack(folder):

    label=folder   

    return label

#------------------------------------------------------------File works
def getDataFiles(TEST_or_Train_TOPfloder):

    return os.listdir(TEST_or_Train_TOPfloder)

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
    return img.astype('float32')

def TFRecord_writer(writer,img,label):
     img=tf.cast(img,tf.uint8)
     label=tf.cast(label,tf.int32)
     img=(img.eval()).tobytes()     
     label=(label.eval()).tobytes()
     example=tf.train.Example(features=
                              tf.train.Features(feature=
                                                {"img":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])), 
                                                 "label":tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
                                                 }
                                                )  
                              )
   
     writer.write(record=example.SerializeToString())
    

#-----------------------------------------------------------------------Image works
def change_img_dim(img):

    b=img.shape[0]
    c=img.shape[-1]

    return np.reshape(img,(b,-1,c))

def change_brightness_hue_randomly(img,change_round=5,clip=None):
    if b==None:
        b=img.shape[0]
    if c==None:
        c=img.shape[-1]
    if clip==None:
        clip=np.random.randint(low=0,high=50)/100
    brt=np.clip(np.random.rand(change_round),-1.*clip,clip).astype('float32')
    hue=np.clip(np.random.rand(change_round),-1.*clip,clip).astype('float32')

   
    for i in range(change_round):
        temp=tf.image.adjust_brightness(img,brt[i])
        temp=tf.image.adjust_hue(img,hue[i])
        if i==0:
            out=temp
        else:
            out = tf.concat((out,temp),0)
      
  
    return out

def change_contrast(img,val=None):
    if b==None:
        b=img.shape[0]
    if c==None:
        c=img.shape[-1]
    if val==None:
        val=np.random.randint(low=0,high=30)/100   
           
    out=tf.image.adjust_contrast(img,val)
    
    return tf.concat((out,img),0)

def flip_it_randomly(img):

                
    sp=np.random.randint(low=0,high=2)      
    if sp==0:
        out=tf.image.random_flip_left_right(img)
    elif sp==1:
        out=tf.image.random_flip_up_down(img)
    else:
        out=tf.image.random_flip_left_right(img)
        out=tf.image.random_flip_up_down(img)

    return tf.concat((out,img),0)

def noisy_img_randomly(img,c=3,weight=2):   #ok
          
    noi=2*np.random.rand(1).astype('float32')
   
    noisy_image=img+noi
    if c==3:
        noisy_image = tf.clip_by_value(noisy_image, 0.0,255.0)
    else:
        noisy_image = tf.clip_by_value(noisy_image,0.0, 1.0)
    
    return  tf.concat((noisy_image,img),0)

def return_to_or_shape(img,img_size,c=None):
           
    b=int(img.shape[0])
    if c==None:
        c=int(img.shape[-1])

    return  tf.reshape(tf.expand_dims(img,1),(b,img_size[0],img_size[1],c))

def crop_randomly(img,imgsize=None,crop_round=6):


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
        if i==0:
            out=temp
        else:
            out=tf.concat((out,temp),0)
   
    return tf.reshape(out,(int(out.shape[0]),crop_side,crop_side,int(out.shape[-1])))

def resize_it(img,new_size):

     return tf.image.resize_images(img,new_size)
 

if __name__=='__main__':

    #writer = tf.python_io.TFRecordWriter('H:/Wtest/saa') 
    with tf.Session() as sess:
        get_all_done('H:/TRUEDATA/B','H:/trueTF',[[1080,1920],1080*1920],128,2,2)
        
        #print('ok')
        #img = input_prepare_pack('H:/Wtest/A/train1/0',[[1024,1024],1024**2],0,0)
        #img=img.eval()
        #img_byte=img.tostring()

        #example=tf.train.Example(
        #features=tf.train.Features(
        #    feature={
        #        "image_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_byte])),                
        #        })  )
   
        #writer.write(record=example.SerializeToString())
        #writer.close()