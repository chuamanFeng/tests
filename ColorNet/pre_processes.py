import numpy as np
import sys
import os
import tensorflow as tf
import PIL.Image as IM


#-------------------------------------------------------------global pack
def get_all_done(main_path,TF_savepath,img_size,file_size,crop_size=[32,32]):

    file_folders_class=getDataFiles(main_path)  

    total_num=0
      
    for j,folder in enumerate(file_folders_class):

        file_num=0
        writer = tf.python_io.TFRecordWriter(TF_savepath+str(folder)+str(file_num)+'.tfrecords')   

        img,img_num=input_prepare_pack(os.path.join(main_path,folder),img_size,crop_size)
        
        total_num+=img_num
     
        if  img_num>file_size:           
            sidx=file_size
            TFRecord_writer(writer,img[0:sidx])
            writer.close()
            file_num+=1            
            img_num-=file_size

            while img_num//file_size>0:
                writer = tf.python_io.TFRecordWriter(TF_savepath+str(folder)+str(file_num)+'.tfrecords')  
                TFRecord_writer(writer,img[sidx:sidx+file_size])
                writer.close()

                sidx+=file_size
                img_num-=file_size
                file_num+=1
               
            writer = tf.python_io.TFRecordWriter(TF_savepath+str(folder)+str(file_num)+'.tfrecords')  
            TFRecord_writer(writer,img[sidx:])
         
        
        else:                
            TFRecord_writer(writer,img)
        writer.close()    
        print(j+1,'folder(s) finished')

    print(total_num)
    writer.close()
  
#-------------------------------------------------------------local folder pack

def input_prepare_pack(filepath,img_size,crop_size=[32,32]):  #for one class

    filenames= getDataFiles(filepath)    #single class_folder
    img_raw=image_to_array(filenames,filepath,img_size)

    img = change_img_dim(img_raw)
    b=img.shape[0]
    c=img.shape[-1]
    
    img = flip_it_randomly(img)  
    img =return_to_or_shape(img,img_size[0],c)
    
    img = crop_it(img,img_size[0],crop_size)
    
    img_num=img.get_shape()[0].value
    return img,img_num



#------------------------------------------------------------File works
def getDataFiles(TEST_or_Train_TOPfloder):

    return os.listdir(TEST_or_Train_TOPfloder)

def image_to_array(filenames,filepath,img_size): #ok

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
    print('transformed')  
    '''                                             #this part have permission error
    os.chdir(savepath) 
    with open(savepath, mode='wb') as f:
        pk.dump(result, f)
    '''    
    return img.astype('float32')

def TFRecord_writer(writer,img):
     img=tf.cast(img,tf.uint8)
    
     img=(img.eval()).tobytes()     
   
     example=tf.train.Example(features=
                              tf.train.Features(feature=
                                                {"img":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),                                                  
                                                 }
                                                )  
                              )
   
     writer.write(record=example.SerializeToString())
    

#-----------------------------------------------------------------------Image works
def change_img_dim(img):

    b=img.shape[0]
    c=img.shape[-1]

    return np.reshape(img,(b,-1,c))



def flip_it_randomly(img,b=None,c=None):

    if b==None:
        b=img.shape[0]
    if c==None:
        c=img.shape[-1]
                
    sp=np.random.randint(low=0,high=2)      
    if sp==0:
        out=tf.image.random_flip_left_right(img)
    elif sp==1:
        out=tf.image.random_flip_up_down(img)
    else:
        out=tf.image.random_flip_left_right(img)
        out=tf.image.random_flip_up_down(img)

    return tf.concat((out,img),0)


def return_to_or_shape(img,img_size,c=None):
           
    b=int(img.shape[0])
    if c==None:
        c=int(img.shape[-1])

    return  tf.reshape(tf.expand_dims(img,1),(b,img_size[0],img_size[1],c))

def crop_it(img,imgsize=None,crop_size=[32,32]):

    if imgsize==None:
        h=img.shape[1]
        w=img.shape[2]
    else:
        h=imgsize[0]
        w=imgsize[1]

    ors=origin_list(imgsize,crop_size)
    num=int(ors.shape[0])
    print(num)
    for i in range(num):      
        orc=ors[i]
        temp=img[:,orc[0]:orc[0]+crop_size[0],orc[1]:orc[1]+crop_size[1],:]        
        if i==0:
            out=temp
        else:
            out=tf.concat((out,temp),0)
   
    return tf.reshape(out,(int(out.shape[0]),crop_size[0],crop_size[1],int(out.shape[-1])))

def origin_list(image_size,crop_size):

    H=image_size[0]
    W=image_size[1]
    h=crop_size[0]
    w=crop_size[1]
    if H<h or W<w:
        print('crop_size too big')
        return None

    h_num=H//h
    w_num=W//w

    idxh=np.expand_dims(np.arange(h_num)*crop_size[0],-1)   #(h_num,1)
    idxw=np.expand_dims(np.arange(w_num)*crop_size[1],-1)   #(w_num,1)
    print(idxh.shape,idxw.shape)
    ori_y=(H-h_num*h)//2
    ori_x=(W-w_num*w)//2

    idxh+=ori_y
    idxw+=ori_x

    idxh=np.tile(np.expand_dims(idxh,1),[1,w_num,1])
    idxw=np.tile(np.expand_dims(idxw,0),[h_num,1,1])  #(h_num,w_num,1)
       
    out=np.concatenate((idxh,idxw),-1)#(h_num,w_num,2)
    out=np.reshape(out,(h_num*w_num,2))
    
    return out
           

if __name__=='__main__':

    #writer = tf.python_io.TFRecordWriter('H:/Wtest/saa') 
    with tf.Session() as sess:
        get_all_done('C:/dataset/B/trainset5','C:/dataset/B/TF',[[512,512],512**2],6272)
        
        print('ok')
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