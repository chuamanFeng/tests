import tensorflow as tf
import numpy as np
import basic_tf


#----------------------------------------------------pre-train model----------------------------------------------#
def placeholder_inputs_pre(batch_size,img_size,channel):

    img_pl=tf.placeholder(shape=[batch_size,img_size[0],img_size[1],channel],dtype=tf.float32)
    label_pl=tf.placeholder(shape=[batch_size],dtype=tf.int32)
   
    return img_pl,label_pl

def brute_classify(img,num_class,is_training,bn_decay):#512->4

    with tf.variable_scope('conv_unit1'):

        out = basic_tf.conv2d(img,16,[3,3],'conv_11',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_11',[2,2],'SAME')

        out = basic_tf.conv2d(out,32,[3,3],'conv_12',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_12',[2,2],'SAME')

        out = basic_tf.conv2d(out,64,[3,3],'conv_13',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_13',[2,2],'SAME')
  
        out = basic_tf.conv2d(out,128,[3,3],'conv_14',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_14',[2,2],'SAME')

        out = basic_tf.conv2d(out,256,[3,3],'conv_15',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_15',[2,2],'SAME')

        out = basic_tf.conv2d(out,512,[3,3],'conv_16',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_16',[2,2],'SAME')

        out = basic_tf.max_pool2d(out,[2,2],'maxpool_17',[2,2],'SAME')
    with tf.variable_scope('conv_unit2'):

        out1 = basic_tf.conv2d(out,1024,[3,3],'conv_21',[1,1],'SAME')       
        out1 = basic_tf.conv2d(out1,512,[1,1],'conv_22',[1,1],'SAME')
        out1 = basic_tf.avg_pool2d(out1,[2,2],'pre_avepool',[2,2],'SAME')
        
    with tf.variable_scope('fully_connected_unit'):

        out2 = tf.reshape(out1,(int(out1._shape[0]),-1))  #b,4096     
        out2 = basic_tf.fully_connected(out2,1024,'fc1')
        out2 = basic_tf.dropout(out2,is_training,'dp1',0.5)

        out2 = basic_tf.fully_connected(out2,128,'fc2')
        out2 = basic_tf.dropout(out2,is_training,'dp2',0.5)

        pred = basic_tf.fully_connected(out2,num_class,'fc3')
    print(pred)
    return pred

def get_loss_pre(predicts, labels):

    with tf.variable_scope('get_loss'):
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predicts,labels=labels)
        loss=tf.reduce_mean(loss,0)
    return loss

#----------------------------------------------------serious model--------------------------------------------------#
def placeholder_inputs(batch_size,img_size,channel,max_object_num):

    img_pl=tf.placeholder(shape=[batch_size,img_size[0],img_size[1],channel],dtype=tf.float32)
    label_pl=tf.placeholder(shape=[batch_size,max_object_num,5],dtype=tf.int32)
    
    return img_pl,label_pl

def basic_detectModel(img,is_training,bn_decay,cell_size,num_class,box_num):#512->4

    with tf.variable_scope('conv_unit1'):

        out = basic_tf.conv2d(img,16,[3,3],'conv_11',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_11',[2,2],'SAME')

        out = basic_tf.conv2d(out,32,[3,3],'conv_12',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_12',[2,2],'SAME')

        out = basic_tf.conv2d(out,64,[3,3],'conv_13',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_13',[2,2],'SAME')
  
        out = basic_tf.conv2d(out,128,[3,3],'conv_14',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_14',[2,2],'SAME')

        out = basic_tf.conv2d(out,256,[3,3],'conv_15',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_15',[2,2],'SAME')

        out = basic_tf.conv2d(out,512,[3,3],'conv_16',[1,1],'SAME')
        out = basic_tf.max_pool2d(out,[2,2],'maxpool_16',[2,2],'SAME')

        out = basic_tf.max_pool2d(out,[2,2],'maxpool_17',[2,2],'SAME')
    with tf.variable_scope('conv_unit2'):

        out1 = basic_tf.conv2d(out,1024,[3,3],'conv_21',[1,1],'SAME')       
        out1 = basic_tf.conv2d(out1,512,[1,1],'conv_22',[1,1],'SAME')      
        
    with tf.variable_scope('fully_connected_unit'):
               
        out2 = tf.reshape(out1,(int(out1._shape[0]),-1))  #b,8192
        out2 = basic_tf.fully_connected(out2,4096,'fc1')
        out2 = basic_tf.dropout(out2,is_training,'dp1',0.5)

        out2 = basic_tf.fully_connected(out2,1024,'fc2')
        out2 = basic_tf.dropout(out2,is_training,'dp2',0.5)

        out2 = basic_tf.fully_connected(out2,cell_size[0]*cell_size[1]*(class_num+box_num*5),'fc3')

    with tf.variable_scope('output_unit'):
        n1 = cell_size[0]*cell_size[1]*num_class    #4*4*2=32
        n2 = n1+cell_size[0]*cell_size[1]*box_num   #32+4*4*2=64

        class_pred= tf.reshape(out2[:,0:n1], (-1, cell_size[0], cell_size[1],num_class))   #(b,4,4,2)
        scales = tf.reshape(out2[:,n1:n2], (-1, cell_size[0], cell_size[1],box_num))       #(b,4,4,2)
        boxes = tf.reshape(out2[:,n2:], (-1,cell_size[0], cell_size[1],box_num*4))         #(b,4,4,8)

        pred = tf.concat([class_pred, scales, boxes], 3)     #(b,4,4,12)
        

    return pred
  

def iou_calc(boxes1, boxes2):
    """calculate ious
    Args:
      boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
    Return:
      iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    with tf.variable_scope('calc_IOU'):
    #calculate the left ups&right downs
        boxes1 = tf.stack([boxes1[:,:,:,0]-boxes1[:,:,:,2]/2,
                           boxes1[:,:,:,1]-boxes1[:,:,:,3]/2,
                           boxes1[:,:,:,0]+boxes1[:,:,:,2]/2,
                           boxes1[:,:,:,1]+boxes1[:,:,:,3]/2])
        boxes1 = tf.cast( tf.transpose(boxes1, [1,2,3,0]),tf.float32)
       #[c1,c2,bo,(yl,xl,yr,xr)]

        boxes2 = tf.cast( tf.stack([boxes2[0]-boxes2[2]/2,
                           boxes2[1]-boxes2[3]/2,
                           boxes2[0]+boxes2[2]/2,
                           boxes2[1]+boxes2[3]/2]),tf.float32)

        #bouding the two
        lu = tf.maximum(boxes1[:,:,:,0:2], boxes2[0:2])   
        rd = tf.minimum(boxes1[:,:,:,2:], boxes2[2:])
        intersection = rd-lu #(dy,dx)

        #intersection
        inter_square = intersection[:,:,:,0]*intersection[:,:,:,1]
        mask = tf.cast((intersection[:,:,:,0]>0), tf.float32)*tf.cast((intersection[:,:,:,1]>0), tf.float32)
    
        inter_square = mask * inter_square  #guarantee its effectiveness
    
        #calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:,:,:,2]-boxes1[:,:,:,0])*(boxes1[:,:,:,3]-boxes1[:,:,:,1])
        square2 = (boxes2[2] - boxes2[0]) * (boxes2[3]-boxes2[1])
    
        #set this as IOU confidence(P)
        iou=tf.truediv(inter_square,(square1 + square2 - inter_square + 1e-6))  #(bo,c1,c2)

    return iou

def loss_calc_loop(object_idx, object_num, loss, predict, labels, image_size,cell_size,num_class,box_num,scales):  
    """
    calculate loss
    Args:
      predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
      labels : [max_objects, 5]  (x_center, y_center, w, h, class)
      scales[class,object,ant_object,coord]
    """
    label = labels[object_idx:object_idx+1, :]
    label = tf.reshape(label, [-1])
    ratio=image_size /cell_size
    
    with tf.variable_scope('calc_objects_tensor'):
    #calculate objects  tensor [CELL_SIZE, CELL_SIZE]  normalize object-window from image_size scale to cell_size scale
    
        min_x = tf.floor((label[0] - label[2] / 2) / ratio)
        max_x = tf.ceil((label[0] + label[2] / 2) / ratio)

        min_y = tf.floor((label[1] - label[3] / 2) / ratio)
        max_y = tf.ceil((label[1] + label[3] / 2) / ratio)   

        temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
        objects = tf.ones(temp, tf.float32)

        temp = tf.cast(tf.stack([min_y, cell_size - max_y, min_x, cell_size - max_x]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        objects = tf.pad(objects, temp, "CONSTANT")

  
    with tf.variable_scope('calc_response_tensor'):
        center_x = label[0] / ratio
        center_x = tf.floor(center_x)

        center_y = label[1] / ratio
        center_y = tf.floor(center_y)

        response = tf.ones([1, 1], tf.float32)

        temp = tf.cast(tf.stack([center_y, cell_size - center_y - 1, center_x, cell_size -center_x - 1]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        response = tf.pad(response, temp, "CONSTANT")
        response = tf.reshape(response, (cell_size,cell_size,1))

    with tf.variable_scope('calc_confidence'):
   
        predict_boxes = predict[:, :, num_class + box_num:]    #predicet:[cell,cell,num_class+box_num+4*box_num]
        predict_boxes = tf.reshape(predict_boxes, (cell_size,cell_size, box_num, 4))
        predict_boxes =tf.multiply( predict_boxes,[ratio, ratio,image_size,image_size]) #back to image scale

        box_origins = np.zeros([cell_size, cell_size, 4]).astype('float32')

        for y in range(cell_size):
          for x in range(cell_size):
           
             box_origins[y, x, :] = [ratio* x, ratio * y, 0, 0]   #(image-scale box origins' coordination)

        box_origins = np.tile(np.resize(box_origins, [cell_size, cell_size, 1, 4]), [1, 1, box_num, 1])
        predict_boxes =tf.add( box_origins ,predict_boxes)   #predicted_boxes back to image-coordination

        iou_predict_truth =iou_calc(predict_boxes, label[0:4])
   
        #[cell_size, cell_size, box]
        confidence = iou_predict_truth * response        
        max_confidence = tf.reduce_max(confidence, 2, True)

        posi_response = tf.cast((confidence >= max_confidence), tf.float32) * response   #high-confidence boxes' responses             
        neg_response = tf.ones_like(confidence, dtype=tf.float32) - confidence      #low-confidence boxes' responses 

    with tf.variable_scope('calc_loss'):

        p_confidence= predict[:,:,num_class:num_class + box_num]    #(boxes' matching-class confidence in each cell)
        p_class = predict[:, :, 0:num_class]
        #class label&predicted class [CELL_SIZE, CELL_SIZE, NUM_CLASSES]       
        label_class= tf.one_hot(tf.cast(label[4], tf.int32), num_class,dtype=tf.float32)                 
        
        #location label&predicted location [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        label=tf.cast(label[0:4],tf.float32)
        x = label[0]       
        y = label[1]
        sqrt_w = tf.sqrt(tf.abs(label[2]))
        sqrt_h = tf.sqrt(tf.abs(label[3]))

        
        p_x = predict_boxes[:, :, :, 0]   #float32      
        p_y = predict_boxes[:, :, :, 1]
        p_sqrt_w = tf.sqrt(tf.minimum(image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(tf.minimum(image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))

        
        #coord_loss(cell-scale)
        coord_loss = (tf.nn.l2_loss(posi_response * (p_x - x)/(ratio)) +
                    tf.nn.l2_loss(posi_response * (p_y - y)/(ratio)) +
                    tf.nn.l2_loss(posi_response * (p_sqrt_w - sqrt_w))/ image_size +
                    tf.nn.l2_loss(posi_response * (p_sqrt_h - sqrt_h))/image_size) * scales[3]

        #class_loss
        class_loss = tf.nn.l2_loss(tf.reshape(objects, (cell_size, cell_size, 1)) * (p_class - label_class)) * scales[0]      
       
        #object_loss
        object_loss = tf.nn.l2_loss( posi_response * (p_confidence - confidence)) * scales[1]      

        #noobject_loss       
        ant_object_loss = tf.nn.l2_loss(neg_response * (p_confidence)) * scales[2]

    return object_idx+ 1, [loss[0] + class_loss, loss[1] + object_loss, loss[2] + ant_object_loss, loss[3] + coord_loss], predict, labels

def get_loss(predicts, labels, objects_num,batch_size,image_size,cell_size,num_class,box_num,scales):
    """Add Loss to all the trainable variables

    Args:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
      ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
      labels  : 3-D tensor of [batch_size, 1, 5]
      objects_num: 1-D tensor [batch_size]  anpother label indeed(0 or 1)
      scales:1-D [4]
    """
    with tf.variable_scope('loss_gather'):
        class_loss = tf.constant(0, tf.float32)
        object_loss = tf.constant(0, tf.float32)
        ant_object_loss = tf.constant(0, tf.float32)
        coord_loss = tf.constant(0, tf.float32)
        loss = [0., 0., 0., 0.]

        for i in range(batch_size):
          predict = predicts[i, :, :, :]
          label = labels[i, :, :]
          object_num = objects_num[i]     
          object_idx=0

          while object_idx<object_num:          
              res =loss_calc_loop(object_idx, object_num, loss, predict, labels, image_size,cell_size,num_class,box_num,scales) 
              object_idx=res[0]             
              loss=res[1]
              predict=res[2]
              labels=res[3]
         
        tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2] + loss[3])/batch_size)

        tf.summary.scalar('class_loss', loss[0]/batch_size)
        tf.summary.scalar('object_loss', loss[1]/batch_size)
        tf.summary.scalar('noobject_loss', loss[2]/batch_size)
        tf.summary.scalar('coord_loss', loss[3]/batch_size)
        tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses')) - (loss[0] + loss[1] + loss[2] + loss[3])/batch_size )

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


if __name__=='__main__':
  
    img=np.random.rand(4,224,224,3).astype('float32')
    #labels1=np.random.randint(low=0,high=2,size=[4,3,1])
    #labels2=np.random.rand(4,3,4).astype('float32')
    #labels=np.concatenate((labels2,labels1),-1)
    img=tf.constant(img)
    #labels=tf.constant(labels)
    #cell_size=[4,4]
    #image_size=512
    class_num=2
    #box_num=2
    pre=brute_classify(img,tf.constant(True),None,class_num)
    #pre=basic_detectModel(img,tf.constant(True),None,cell_size,class_num,box_num)
    #res=get_loss(pre,labels,[2,2,3,1],4,512,cell_size[0],class_num,box_num,[1,1,1,1])

    print(pre)
  