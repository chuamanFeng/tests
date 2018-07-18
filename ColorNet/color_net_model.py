import tensorflow as tf
import basic_module as bm
import time
import numpy as np
import basic_tf





list_of_mlplist=[[32,32,64],[16,16,32]]
list_of_kernel=[[3,3],[5,5]]

def placeholder_inputs(b,h,w):
    rgb_pl = tf.placeholder(tf.float32, shape=(b,h,w,3))
    srgb_pl = tf.placeholder(tf.float32, shape=(b,h,w,3))
    
    return rgb_pl, srgb_pl


def color_net(rgb,is_training, bn_decay=None):

    with tf.variable_scope('input_layer'):
        h=rgb.get_shape()[1].value
        w=rgb.get_shape()[2].value
        b=rgb.get_shape()[0].value
          
        og=[h,w]          
        end_data={}
        end_data['rgb_data'] =rgb
        #end_data['srgb_data']=srgb    #both of them are been normalized     
        out1=basic_tf.conv2d(rgb,96,[1,1],'input_conv',[1,1],'SAME')
        out1=basic_tf.max_pool2d(out1,[2,2],'input_pool',[1,1],'SAME')

    with tf.variable_scope('intermidate_layer'):
        for i,kernels in enumerate(list_of_kernel):
            mlps=list_of_mlplist[i]     
            out1=bm.ssc_color_info_abstraction(out1,mlps,is_training=is_training, bn_decay=bn_decay, 
                                                scope='ssc_section_%d'%(i), kernel_size=kernels,bn=True
                                              )
            if i==0:
                hyper_colume=out1
            else:
                hyper_colume=tf.concat([hyper_colume,out1],-1)
               
        hyper_colume=basic_tf.avg_pool2d(hyper_colume,[2,2],'medium_avepool',[1,1],'SAME')            
        c=hyper_colume.get_shape()[-1].value
        print(hyper_colume.shape)
        hyper_colume=tf.reshape(hyper_colume,(b*h*w,c))
        
    with tf.variable_scope('output_layer'):  
                
        out = basic_tf.fully_connected(hyper_colume,256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        out = basic_tf.dropout(out, keep_prob=0.5, is_training=is_training, scope='dp2')
        out = basic_tf.fully_connected(out,64, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
        out = basic_tf.dropout(out, keep_prob=0.5, is_training=is_training, scope='dp3')
        out = basic_tf.fully_connected(out,3, bn=True, is_training=is_training, scope='fc4', bn_decay=bn_decay)   

        pred=tf.reshape(out,(b,h,w,3))      
      
    return pred,end_data

#*******************************************************************************#

def get_loss(pred,srgb,end_data):  

    with tf.variable_scope('loss_calc'): 
        b=int(pred.shape[0])    
        pred=tf.reshape(tf.reduce_sum(((pred-srgb)**2),-1),(b,-1))#(b,h*w)
        loss=tf.reduce_mean(pred,-1)  #(b,h*w)->(b)
        loss=tf.reduce_mean(loss,0)
   
    return loss

def error_val(rgb,srgb):
    
    with tf.variable_scope('lab_accuracy_calc'):    
        b=rgb.shape[0]

        rgb=deprocess(rgb)
        srgb=deprocess(srgb)

        Labr=bm.srgb2lab(rgb)
        Labs=bm.srgb2lab(srgb)   #(b,h*w,3)
  
   
        color_error=tf.reduce_sum(((Labr-Labs)**2),-1)  #((b,w*h) 
        color_error=tf.reduce_mean(color_error,-1) #(b,1)
        accuracy=tf.reduce_mean(color_error,0)

  

    return accuracy

def deprocess(img_batch):
   
    return tf.clip_by_value(img_batch * 255., 0., 255.)
  


if __name__=='__main__':

   
    with tf.Session() as sess:
        data=np.random.rand(32,32,32,3).astype('float32')
        data=tf.constant(data)
        res=color_net(data,tf.constant(True))