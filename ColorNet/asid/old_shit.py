#def train_one_epoch(sess, ops, train_writer,trainfolder_list): #one epoch:big_batch_num*mini_batch_num loops,okay
#    """ ops: dict mapping from string to tf ops """
#    is_training = True    

#    rgb=data_list[0]
#    srgb=data_list[1]
    
#    #Shuffle train files
#    b1=rgb.shape[0]    
#    train_batch_idxs = np.arange(0,b1)  #make idxs for files
#    np.random.shuffle(train_batch_idxs)  #shuffle the dim "b1"
#    rgb=rgb[train_batch_idxs]
#    srgb=srgb[train_batch_idxs]
   
#    loss_sum = 0.      
#    total_seen = 0   
#    #every train file is a big batch,we shuffle the idxs of them before
#    for bi in range(b1):   #()
#        #notice
#        log_string('----' + 'train-batch-count='+str(bi) + '-----')
      
#        current_rgb=rgb[bi]          
#        current_srgb=srgb[bi]       
   
#        [current_rgb,current_srgb] = hp.shuffle_it([current_rgb,current_srgb])  #shuffle the dim "B"       
       
                   
#        #feeding 
#        feed_dict = {ops['rgb_pl']: current_rgb,        
#                        ops['srgb_pl']: current_srgb,     
#                        ops['is_training_pl']: is_training,}

#        #run the session and get all the results we need
#        summary, step,_,accuracy,loss_val= sess.run([ops['merged'], ops['step'], 
#            ops['train_op'],ops['accuracy'],ops['loss']], feed_dict=feed_dict)

#        #then cook
#        '''
#        train_writer.add_summary(summary, step)
#        '''   
#        print('accuracy(train):',accuracy)
#        print('loss(train):',loss_val)
#        total_seen += BATCH_SIZE
#        loss_sum += loss_val
   
#    log_string('mean loss of epoch: %f' % (loss_sum / float(b1)))  #()
#    log_string('mean loss of every pair: %f' % (loss_sum / float(total_seen)))  #()


##************************************************test unit*******************************************************    
#def test_one_epoch(sess, ops, test_writer,testfolder_list):
#    """ ops: dict mapping from string to tf ops """
#    """everything is similar to training process,
#       the difference is:
#         a.when we testing we don't need data augment(rotate&jitter);
#         b.we have an extra loop after getiing results from running the session"""
#    is_training = False

#    total_seen = 0
#    total_ave_error =0.
#    loss_sum = 0.

#    rgb=test_data_list[0]
#    srgb=test_data_list[1]
#    b1=rgb.shape[0]  
#    #get some stuff from test-data files,
#    for bi in range(b1):
#        log_string('----'+'test-batch-count='+ str(bi) + '-----')

#        current_rgb=rgb[bi]          
#        current_srgb=srgb[bi]
               

#         #feeding 
#        feed_dict = {ops['rgb_pl']: current_rgb,                
#                     ops['srgb_pl']: current_srgb,      
#                     ops['is_training_pl']: is_training,}

#        #run the session and get all the results we need
#        summary, step,accuracy,loss_val= sess.run([ops['merged'], ops['step'], 
#            ops['accuracy'],ops['loss']], feed_dict=feed_dict)

             
#        print('accuracy(test):',accuracy)
#        print('loss(test):',loss_val)

#        total_ave_error+= accuracy
#        total_seen += BATCH_SIZE
#        loss_sum += (loss_val)
#        #extra loop:
#        #count how many times each class has occured in prediction,and numbering the correct ones
#    log_string('average evaled error of every batch in one epoch:%f' %(total_ave_error/ float(total_seen)))
#    log_string('test mean loss every pair: %f' % (loss_sum / float(total_seen)))    
   

#def color_net1(rgb,is_training, bn_decay=None):

#    with tf.variable_scope('input_layer') as sc1:
#        h=rgb.get_shape()[1].value
#        w=rgb.get_shape()[2].value
#        b=rgb.get_shape()[0].value
   
#        hyper_list=[]    
#        og=[h,w]          
#        end_data={}
#        end_data['rgb_data'] =rgb
#        #end_data['srgb_data']=srgb    #both of them are been normalized
       


#    with tf.variable_scope('intermidate_layer') as sc2:
#        for i,kernels in enumerate(list_of_kernel):
#            mlps=list_of_mlplist[i]     
#            rgb=bm.ssc_color_info_abstraction(rgb,mlps,is_training=is_training, bn_decay=bn_decay, 
#                                                scope='conv_%d'%(i), kernel_size=kernels,bn=True,seperated_kernel=seperated_kernel,
#                                                pooling=pooling,pooling_kernel=pooling_kernel)
#            hyper_list.append(rgb)

#        hyper_colume=bm.down_sample_process(hyper_list,orig_size=og,scope='hyper_trans')
#        c=hyper_colume.get_shape()[-1].value
#        hyper_colume=tf.reshape(hyper_colume,(b*h*w,c))
   
#        out = basic_tf.fully_connected(hyper_colume,512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
#        out = basic_tf.dropout(out, keep_prob=0.8, is_training=is_training, scope='dp1')
#        out = basic_tf.fully_connected(out,256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
#        out = basic_tf.dropout(out, keep_prob=0.8, is_training=is_training, scope='dp2')
#        out = basic_tf.fully_connected(out,128, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
#        out = basic_tf.dropout(out, keep_prob=0.5, is_training=is_training, scope='dp3')
#        out = basic_tf.fully_connected(out,3, bn=True, is_training=is_training, scope='fc4', bn_decay=bn_decay)       

#    with tf.variable_scope('output_layer') as sc3:  
   
#        pred=tf.reshape(out,(b,h,w,3))      
      
#    return pred,end_data


##************************************************************************************#
#def down_sample_process(list_of_hypers,orig_size,scope):
#     '''
#input:
#    list of different scale of info
#output:
#    a tensor of original size images(except the channel num)
#     '''

#     if  list_of_hypers[0].get_shape()[-2].value>list_of_hypers[1].get_shape()[-2].value:   #from small to big size
#         list_of_hypers.reverse()

#     with tf.variable_scope(scope) as sc:  
#         num_hy=len(list_of_hypers)
#         hyper=list_of_hypers[0]
         
#         for i in range(num_hy-1):      
#             hyper2=list_of_hypers[i+1] 
#             size=[hyper2.get_shape()[1].value,hyper2.get_shape()[2].value]  

#             hyper=tf.concat([up_scale(hyper,size),hyper2],-1)
             
#         #hyer size(b,hmax,wmax,sum(mlpi))
#         hyper=up_scale(hyper,orig_size)
#             #hyer1 final size(b,h,w,c=first one mlp_list[-1])
#     print('got hyper_colume ,size',hyper.get_shape())      
#     return hyper

#def up_scale(img,size):
        
#    return (tf.image.resize_images(img,size,0))   
def pre_process_unit(img,crop,orig,dsize):
    '''
input:
    one  rgb/srgb img,as numpy array
output:
    as input,with new size&area maybe
    ''' 
    if img.dtype == np.uint8:       
        img =img/255.0
       
    if crop:        
        h=img.shape[1]
        w=img.shape[2]
        short_edge = min(h,w)
        if dsize[0]>w:
            dsize[0]=w
            print('set original data width as crop width')
        if dsize[1]>h:
            dsize[1]=h
            print('set original data height as crop height')
        if dsize==None:
            dsize=[short_edge,short_edge]
            print('set original data size as crop size')
        
        if random_crop: 
            print('choose origin randomly ')
            yy = np.random.randint(0,h - dsize[1]+1)
            xx = np.random.randint(0,w - dsize[0]+1)  
            orig=[xx,yy]        
      
    return img,orig


def pre_norm_process(rgb,srgb):
    '''
input:
    a list of rgb&srgb dataset(2*(b,h,w,3))
output:
    a list of aver_normed rgb&srgb dataset(2*(b,h,w,3))
    '''  
    b=rgb.shape[0]   
   
    aver_rgb=np.sum(rgb,axis=0)/float(b)
    aver_srgb=np.sum(srgb,axis=0)/float(b)   #(w,h,3)
   
    rgb=np.array(rgb-np.tile(np.expand_dims(aver_rgb,0),[b,1,1,1]))
    srgb=np.array(srgb-np.tile(np.expand_dims(aver_srgb,0),[b,1,1,1]))
    print(rgb.shape)
    data_out=[]
    data_out.append(rgb)
    data_out.append(srgb)

    return data_out   #prob_occur
def seg_into_batches(rgb,srgb,batch_size,auto_arange=True):
      
    b=batch_size
    B=rgb.shape[0]  
    print(B,'*2 PICS for seg')
    h=rgb.shape[1]
    b1=B//b  

    rgb_d=np.concatenate((rgb,srgb),-1)
    if b1>0:
        print('at least seg into',b1,'group(s)')
        w=b1*b
   
        data=rgb_d[:w,:,:,:]          
        data=np.reshape(np.expand_dims(data,0),(b1,b,h,-1,6))
        print(data.shape)
   
        if B%b!=0 :
            if B-b>5:
                if auto_arange:
                    print('seg into',b1+1,'groups of size',batch_size)
                    b1r=B-w
                    res=rgb_d[w:,:,:]
             
                    if b1r<(b/2):
                        b1r*=2
                        res=np.concatenate((res,res),0)
                  

                    need=b-b1r            
                    temp=rgb_d[:w,:,:,:]
              
                    idx=np.arange(w)
                    np.random.shuffle(idx)
                    temp=temp[idx]
            

                    chosen=temp[:need,:,:,:]              
                    data=np.concatenate((data,np.expand_dims(np.concatenate((chosen,res),0),0)),0)    #b1+1    
                
            else:               
                print('seg into',b1,'groups of size',batch_size)

    elif b1==0 and auto_arange:
         print('padding into 1 group')        
         if B<(b/2):
             need=b-2*B
             data=np.tile(rgb_d,[2,1,1,1])            
         else:
             need=b-B    
             data=rgb_d
             
         idx=np.arange(B)
         np.random.shuffle(idx)

         temp=(rgb_d[idx])[:need,:,:,:]  
         print(temp.shape,data.shape)      
         data=np.concatenate((data,temp),0)
         data=np.expand_dims(data,0)

    data=np.array(data).astype('float32')   
   
    rgb_out=data[:,:,:,:,:3]
    srgb_out=data[:,:,:,:,3:]
   
    return rgb_out,srgb_out
