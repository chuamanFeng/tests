import numpy as np
import sys
import os
import PIL.Image as IM
import pickle as pk
import matplotlib.pyplot as pyplot
import tensorflow as tf

save_dir='C:\dataset\B'
data_dir='C:/dataset/A/train/ok'
data_dir2='C:/dataset/A/train/2'
IMG_FORM='.bmp'

def getDataFiles(TEST_or_Train_TOPfloder):

    return os.listdir(TEST_or_Train_TOPfloder)
def get_ImageQueue(child_file,img_form):
    fq=tf.train.string_input_producer(tf.train.match_filenames_once(child_file))  
    image_reader=tf.WholeFileReader()
    _,image_file=image_reader.read(fq)
    if img_form=='bmp' or 'BMP':
        image=tf.image.decode_bmp(image_file)
    if img_form=='jpeg' or 'jpg' or'JPG' or'JPEG' :
        image=tf.image.decode_jpeg(image_file)
    if img_form=='gif'or 'GIF':
        image=tf.image.decode_gif(image_file)
    if img_form =='png'or 'PNG':
        image=tf.image.decode_png(image_file)
    else:
        image=tf.image.decode_image(image_file)
    
    #os.chdir(base)
    return image

def pre_process_before_training(rgb_srgb_list,crop,crop_size,crop_round,crop_origin_list,start_or_idx,or_num=None):
   
    
    N=rgb_srgb_list[0].shape[0]   
    batch_out=N*crop_round   #(6)
    if or_num==None:
        or_num=crop_origin_list.get_shape()[0].value
    if crop:
        end_or_idx=start_or_idx+crop_round
        if end_or_idx>or_num:
            plus=end_or_idx-or_num
            crop_origin_list=crop_origin_list[start_or_idx-plus:end_or_idx-plus,:]
        else:
            crop_origin_list=crop_origin_list[start_or_idx:end_or_idx,:]
        
        for i in range(crop_round):
            origin=crop_origin_list[i]
            out=batch_pre_crop(rgb_srgb_list,crop,crop_size,origin)  #2*(b,h',w',3)
            out1=out[0]
            out2=out[1]

            if i==0 :
                out_rgb=out1
                out_srgb=out2
            else:
               
                out_rgb=np.concatenate((out_rgb,out1),0)
                out_srgb=np.concatenate((out_srgb,out2),0)
    else:
        out_rgb=rgb_srgb_list[0]
        out_srgb=rgb_srgb_list[1]
   

    out_data=[out_rgb,out_srgb]
    return out_data,end_or_idx

def batch_pre_crop(rgb_srgb_list,crop,crop_size,crop_origin):
   '''
input:
   a list of 2 (b,h,w,3)array
   
   '''
   boo=check_pair_dims(rgb_srgb_list)
   if boo==False:
       print('___please check the size of data_list,this sucks with two shapes___')
       return None

   rgb=rgb_srgb_list[0] 
   srgb=rgb_srgb_list[1]
   #croping 
   crop_rgb=rgb[:,crop_origin[0]:crop_origin[0]+crop_size[0],crop_origin[1]:crop_origin[1]+crop_size[1],:]  
   crop_srgb=srgb[:,crop_origin[0]:crop_origin[0]+crop_size[0],crop_origin[1]:crop_origin[1]+crop_size[1],:]    

   #average 
   data_out=[crop_rgb,crop_srgb]
   boo1=check_pair_dims(data_out)
   if boo1==False:
       print('___please check the size of out_data,this sucks with two shapes___')
       return None

   return data_out

def check_pair_dims(pair_list):

  
    if pair_list[0].shape==pair_list[1].shape:
        return True
    else:
        return False



def file_filter(filepath=data_dir,img_form=IMG_FORM):
    filenames=os.listdir(filepath)
    effective_files=[]
    for name in filenames:
        if os.path.splitext(name)[1]==img_form:
            effective_files.append(name)

    return effective_files

def image_to_array(filenames,filepath,start_idx,batch_size,img_size,img_form=IMG_FORM):

    name_num=int(len(filenames))
   
    h=img_size[0][0]
    w=img_size[0][1]
    area=img_size[1]
 
    result = np.array([])  
    end_idx=start_idx+batch_size
  
    if end_idx<=name_num:
        filenames=filenames[start_idx:end_idx]
      
    else:
        plus=end_idx-name_num
        filenames=filenames[start_idx-plus:name_num]          
  
    print('transforming...')
    for name in filenames:
        image = IM.open(os.path.join(filepath,name))
        r, g, b = image.split() 
      
        r_arr = np.array(r).reshape(area)
        g_arr = np.array(g).reshape(area)
        b_arr = np.array(b).reshape(area)
       
        image_arr = np.concatenate((r_arr, g_arr, b_arr))
        result = np.concatenate((result, image_arr))

    result = np.reshape(result,(-1,h,w,3)) 
   
    print('transformed')  
    '''                                             #this part have permission error
    os.chdir(savepath) 
    with open(savepath, mode='wb') as f:
        pk.dump(result, f)

    ''' 
    result=pre_process(result)  
    return result,end_idx

def array_to_image(data_arr,filepath=save_dir,depros=False):#ok
     
    base=os.getcwd()
    os.chdir(filepath)  

   
    b = data_arr.shape[0]
    arr = data_arr.reshape(b,3,IMAGE_HEIGHT,IMAGE_WIDTH)
    for i in range(b):
        a = arr[i]
          
        r = IM.fromarray(a[0]).convert('L')
        g = IM.fromarray(a[1]).convert('L')
        b = IM.fromarray(a[2]).convert('L')
        image = IM.merge("RGB", (r, g, b))
        if depros:
            image=deprocess(image)
        image.save(os.path.join(filepath ,("result" +str(i)+ ".png")), 'png')

    os.chdir(base)
    print('finished')

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
   
    return out
           

def pre_process(img):

    if int(img.shape[-1])==3:
        return img/255.0
    else:
        return img/1.0

def deprocess(img,color_norm=True):
    if color_norm:
        return np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        return np.clip(img,0,1).astype(np.unit8)

def shuffle_it(data_list,b=None):
       
    if b==None:
        b=data_list[0].shape[0]
    idx=np.arange(b)
    np.random.shuffle(idx)
    
    data0=data_list[0][idx]
    data1=data_list[1][idx]

    return [data0,data1]
def shuffle_it_sin(input,b=None):
     if b==None:
        b=input.shape[0]
     idx=np.arange(b)
     np.random.shuffle(idx)

     return input[idx]

if __name__=='__main__':

  
    result1=image_to_array(data_dir,[1024,1024],img_form=IMG_FORM)
    result2=image_to_array(data_dir2,[1024,1024],img_form=IMG_FORM)
    result=[result1,result2]
    out= pre_process_before_training(result,32,crop=True,crop_size=[32,32],crop_round=3,random_crop=True)
    print(out[0].shape)
    #srgb2lab(data)
    #list=seg_into_batches(result1,result2,32)