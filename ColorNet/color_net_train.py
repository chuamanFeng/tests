import argparse
import math
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
import basic_tf
import pickle as pk
import color_net_model 
import basic_module as bm
import tf_dataLoader as tl

#----------------------------------make a parsel for argments infos:  250 epoch,nearly 5 hours---------------------------#
parser = argparse.ArgumentParser()

parser.add_argument('--model', default='color_net_model', help='Model name')
parser.add_argument('--log_dir', default='C:\dataset', help='Log dir [default: log]')
parser.add_argument('--crop_size', default=[32,32], help='lenth&width for croping')
parser.add_argument('--max_epoch', type=int, default=150, help='Epoch to run [default: 150]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

#------------------------------------get those arguments from the parsel---------------------------------------------------#
BATCH_SIZE = FLAGS.batch_size
CROP_SIZE = FLAGS.crop_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


MODEL = importlib.import_module(FLAGS.model) # import network module
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): 
    os.mkdir(LOG_DIR)

    '''
    some sort of copy work
os.system('COPY %s %s' % ( FLAGS.model+'.py', LOG_DIR)) # bkp of model def
os.system('COPY color_net_train.py %s' % (LOG_DIR)) # bkp of train procedure
    '''
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

#-------------------------------------other params,including hyper params & data address----------------------------------------------------#
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

IMG_FORM='bmp'
IMAGE_HEIGHT=1024
IMAGE_WIDTH=1024
IMAGE_SIZE=[IMAGE_HEIGHT,IMAGE_WIDTH]
IMAGE_AREA=IMAGE_HEIGHT*IMAGE_WIDTH
CHANNEL=3

data_dir='C:/dataset/A'
data_dir_test='C:/dataset/C'
data_dir_vali='C:/dataset/B'
HOSTNAME = socket.gethostname()

TRAIN_FILES =os.listdir(data_dir)
FILE_LENTH_TRAIN=int(len(TRAIN_FILES))
TEST_FILES =os.listdir(data_dir_test)
FILE_LENTH_TEST=int(len(TEST_FILES))
VALI_FILES =os.listdir(data_dir_vali)
FILE_LENTH_VALI=int(len(VALI_FILES))
FULL_FILE_LIST=tl.full_path_maker([data_dir,data_dir_test,data_dir_vali],[TRAIN_FILES,TEST_FILES,VALI_FILES])
#-----------------------------------------basic helpers for retrieving learnable params------------------------------------------------------#
#string helpler
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

#learning rate(exponential)
def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate       
 
#batch_norm_decay(exponential)
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay
#---------------------------------------------------------------------------------------------------#
def train(trainfile_lenth=FILE_LENTH_TRAIN,testfile_lenth=FILE_LENTH_TEST,valifile_lenth=FILE_LENTH_VALI): 
    '''
 elements:
   A part
    placeholder for inputs
    batch&batch_normalization decay
    model&its loss/accuracy
    optimizer(train_op)
    saver
   B part
    session
    summary writer(train&test)
    global variable initializer(use that sess run it before training)
    operations in a dictionary
   C part
    a loop for training (train&eval&summary)
    '''
    with tf.Graph().as_default():

        with tf.variable_scope('input_placeholders'):
            rgb_pl, srgb_pl= MODEL.placeholder_inputs(BATCH_SIZE,CROP_SIZE[0],CROP_SIZE[1])
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
        # Note the global_step=batch parameter to minimize. 
        # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
        with tf.variable_scope('scalars'):
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss and eval
            pred,end_data = MODEL.color_net(rgb_pl, is_training_pl,bn_decay=bn_decay)
            loss = MODEL.get_loss(pred,srgb_pl,end_data)
            accuracy=MODEL.error_val(pred,srgb_pl)
            tf.summary.scalar('loss', loss)
         
            # Get optimizer
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
        
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            with tf.variable_scope('data_packs_ops'):
                # data_pip_line           
                ops_traindata={}
                ops_testdata={}
                ops_validata={}
                for i ,path in enumerate(FULL_FILE_LIST[0]):
                    path_rgb=path+'/rgb'
                    path_srgb=path+'/srgb'
                    
                    list_rgb,list_srgb=tl.TF_loader_multi(path_rgb,path_srgb,[CROP_SIZE[0],CROP_SIZE[1],CHANNEL],num_epochs=None)
               
                    ops_traindata['list_rgb_%s'%(i)]=list_rgb
                    ops_traindata['list_srgb_%s'%(i)]=list_srgb

                for i,path in enumerate(FULL_FILE_LIST[1]):
                    path_rgb=path+'/rgb'
                    path_srgb=path+'/srgb'

                    list_rgb,list_srgb=tl.TF_loader_multi(path_rgb,path_srgb,[CROP_SIZE[0],CROP_SIZE[1],CHANNEL],num_epochs=None)
               
                    ops_testdata['list_rgb_%s'%(i)]=list_rgb
                    ops_testdata['list_srgb_%s'%(i)]=list_srgb

                for i,path in enumerate(FULL_FILE_LIST[2]):
                    path_rgb=path+'/rgb'
                    path_srgb=path+'/srgb'

                    list_rgb,list_srgb=tl.TF_loader_multi(path_rgb,path_srgb,[CROP_SIZE[0],CROP_SIZE[1],CHANNEL],num_epochs=None)
               
                    ops_validata['list_rgb_%s'%(i)]=list_rgb
                    ops_validata['list_srgb_%s'%(i)]=list_srgb

        with tf.variable_scope('training_session'):    
        # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True #allow grouing gpu usage
            config.allow_soft_placement = True   #allow tf  charge the devices when our charge failed
            config.log_device_placement = False  #print device configuration infos
            sess = tf.Session(config=config)
        
            # Add summary writers    
            merged = tf.summary.merge_all()    
       
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train_color'),
                                      sess.graph)
            eval_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'eval_color'),
                                               sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test_color'))             
      
             # Init variables
            init = (tf.global_variables_initializer() ,tf.local_variables_initializer()) 
               
            sess.run(init, {is_training_pl: True})  #kind of slow but passible
        
            ops = {'rgb_pl': rgb_pl,
                   'srgb_pl': srgb_pl,
                   'is_training_pl': is_training_pl,
                   'pred':pred,
                   'accuracy':accuracy,                          
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch}
            #check if we have a trained model(check_point)
            initial_step=0
            ckpt=tf.train.get_checkpoint_state(LOG_DIR)     
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)   
                print('will be started from the last step')        
                initial_step=int(ckpt.model_checkpoint_path.rsplit('-',1)[1])
    
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                while not coord.should_stop():    
                    for epoch in range(initial_step,MAX_EPOCH):
                        log_string('*******\\\\\\\**************** EPOCH %03d ***************///////*******' % (epoch))
                        sys.stdout.flush()
                                     
                        print('________^@^___training&evaluating_one_epoch___^@^________') 
                        train_one_epoch(sess,ops,ops_traindata,train_writer,trainfile_lenth)
                        testOReval_one_epoch(sess, ops,ops_validata,eval_writer,valifile_lenth)

                        if epoch % 5 == 0 and epoch!=0:
                            print('________^@^___testing_for_every_five_epochs___^@^________ ')
                            testOReval_one_epoch(sess, ops,ops_testdata, test_writer,testfile_lenth)
                       
                        if epoch % 10 == 0 :
                            save_path = saver.save(sess, os.path.join(LOG_DIR, "model-"+str(epoch)))
                            log_string("Model saved in file: %s" % save_path)
            except tf.errors.OutOfRangeError:
                print ('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()



#************************************************train unit*******************************************************

def train_one_epoch(sess, ops,ops_data, train_writer,file_lenth,batch_size=BATCH_SIZE):


    is_training = True    
    loss_sum = 0.      
    total_seen = 0    
    cnt_loop=0
    for i in range(file_lenth):  #tfrecords   
        with tf.variable_scope('data_packer'):
            list_rgb,list_srgb=sess.run([ops_data['list_rgb_%s'%(i)],ops_data['list_srgb_%s'%(i)]])
            current_rgb,current_srgb=tl.data_loader(list_rgb,list_srgb)
       
        idx=0
        num=int(current_rgb.shape[0])
        
        with tf.variable_scope('train_layer') as sc:    
            while idx<num:              
                log_string('----' + 'train-batch-count='+str(cnt_loop) + '-----')                   
                rgb=current_rgb[idx:idx+batch_size]
                srgb=current_srgb[idx:idx+batch_size]
             
                idx+=batch_size                   
       
                   
                #feeding 
                feed_dict = {ops['rgb_pl']: rgb,        
                                ops['srgb_pl']: srgb,     
                                ops['is_training_pl']: is_training,}

                #run the session and get all the results we need
                summary, step,_,accuracy,loss_val= sess.run([ops['merged'], ops['step'], 
                    ops['train_op'],ops['accuracy'],ops['loss']], feed_dict=feed_dict)

                #then cook               
                train_writer.add_summary(summary, step)
                                  
                print('accuracy(train):',accuracy)
                print('loss(train):',loss_val)

                total_seen += BATCH_SIZE
                loss_sum += loss_val
                cnt_loop+=1
                            
    log_string('mean loss/pair: %f' % (loss_sum / float(cnt_loop)))  #()
 
def testOReval_one_epoch(sess, ops,ops_data, testOReval_writer,file_lenth,batch_size=BATCH_SIZE):


    is_training = False   
    loss_sum = 0.      
    total_seen = 0    
    total_ave_error =0.
    cnt_loop=0
    for i in range(file_lenth):  #tfrecords   
        with tf.variable_scope('data_packer'):
            list_rgb,list_srgb=sess.run([ops_data['list_rgb_%s'%(i)],ops_data['list_srgb_%s'%(i)]])
            current_rgb,current_srgb=tl.data_loader(list_rgb,list_srgb)
       
        idx=0
        num=int(current_rgb.shape[0])
        
        with tf.variable_scope('testORvali_layer') as sc:    
            while idx<num:              
                log_string('----' + 'testORvali-batch-count='+str(cnt_loop) + '-----')                   
                rgb=current_rgb[idx:idx+batch_size]
                srgb=current_srgb[idx:idx+batch_size]
             
                idx+=batch_size                   
       
                   
                #feeding 
                feed_dict = {ops['rgb_pl']: rgb,        
                                ops['srgb_pl']: srgb,     
                                ops['is_training_pl']: is_training,}

                #run the session and get all the results we need
                summary, step,accuracy,loss_val= sess.run([ops['merged'], ops['step'], 
                    ops['accuracy'],ops['loss']], feed_dict=feed_dict)

                #then cook               
                testORval_writer.add_summary(summary, step)
                
                print('accuracy(test):',accuracy)
                print('loss(test):',loss_val)

                total_ave_error+= accuracy
                total_seen += BATCH_SIZE
                loss_sum += (loss_val)
                cnt_loop+=1

    log_string('mean Lab-accuracy/pair:%f' %(total_ave_error/ float(cnt_loop)))                        
    log_string('mean loss/pair: %f' % (loss_sum / float(cnt_loop)))  #()   
        



if __name__=="__main__":
       
    train()
    LOG_FOUT.close()
  