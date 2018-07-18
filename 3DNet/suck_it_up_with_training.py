import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
import tf_dataLoader as tl
import basic_tf
import pickle as pk

#----------------------------------make a parsel for argments infos:  250 epoch,nearly 5 hours---------------------------#
parser = argparse.ArgumentParser()
#parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='CL_modules', help='Model name: CL_modules or CLftSG_modules [default: CL_modules]')
parser.add_argument('--log_dir', default='H:\D3test', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=102400, help='Point Number[default: 102400]')
parser.add_argument('--max_epoch', type=int, default=150, help='Epoch to run [default: 150]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

#------------------------------------get those arguments from the parsel---------------------------------------------------#

#GPU_INDEX = FLAGS.gpu
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): 
    os.mkdir(LOG_DIR)
#os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
#os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


#-------------------------------------other params,including hyper params & data address----------------------------------------------------#
MAX_NUM_POINT = 204800
NUM_CLASSES = 3

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()
CHANNEL=3
save_dir='H:\D3test'
data_dir='H:\D3test\A'   #HERE WE GONNA USE TFRecords
data_dir_test='H:\D3test\B'
data_dir_vali='H:\D3test\C'
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


#-----------------------------------------------------------training-----------------------------------------------------------#
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
      
        with tf.variable_scope('input_placeholder'):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
        with tf.variable_scope('scalars'):
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model_single_cl(pointclouds_pl,NUM_CLASSES,BATCH_SIZE, is_training_pl,bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)
        
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)
        
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
                list_cloud,list_label=tl.TF_loader_multi(path,[NUM_POINT,CHANNEL+3],num_epochs=None)
               
                ops_traindata['list_cloud_%s'%(i)]=list_cloud
                ops_traindata['list_label_%s'%(i)]=list_label

            for i,path in enumerate(FULL_FILE_LIST[1]):
                list_cloud,list_label=tl.TF_loader_multi(path,[NUM_POINT,CHANNEL+3],num_epochs=None)
               
                ops_testdata['list_cloud_%s'%(i)]=list_cloud
                ops_testdata['list_label_%s'%(i)]=list_label

            for i,path in enumerate(FULL_FILE_LIST[2]):
                list_cloud,list_label=tl.TF_loader_multi(path,[NUM_POINT,CHANNEL+3],num_epochs=None)
               
                ops_validata['list_cloud_%s'%(i)]=list_cloud
                ops_validata['list_label_%s'%(i)]=list_label

        with tf.variable_scope('training_session'):   
            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True #allow grouing gpu usage
            config.allow_soft_placement = True   #allow tf  charge the devices when our charge failed
            config.log_device_placement = False  #print device configuration infos
            sess = tf.Session(config=config)
        
            # Add summary writers    
            merged = tf.summary.merge_all()    
       
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train_3D'),
                                      sess.graph)
            eval_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'eval_3D'),
                                               sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test_3D'),sess.graph)             
      
             # Init variables
            init = (tf.global_variables_initializer(), tf.local_variables_initializer())
               
            sess.run(init, {is_training_pl: True})  #kind of slow but passible
           
            ops = {'pointclouds_pl': pointclouds_pl,
                   'labels_pl': labels_pl,
                   'is_training_pl': is_training_pl,
                   'pred': pred,
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

def train_one_epoch(sess,ops,ops_data,train_writer,file_lenth,batch_size=BATCH_SIZE):

    is_training = True    
    loss_sum = 0.      
    total_seen = 0  
    total_correct=0.  
    cnt_loop=0 

    for i in range(file_lenth):  #tfrecords   
        with tf.variable_scope('data_packer'):
            list_cloud,list_label=sess.run([ops_data['list_cloud_%s'%(i)],ops_data['list_label_%s'%(i)]])
            current_cloud,current_label=tl.data_loader(list_cloud,list_label)
       
        idx=0
        num=int(current_cloud.shape[0])
        
        with tf.variable_scope('train_layer') as sc:    
            while idx<num:              
                log_string('----' + 'train-batch-count='+str(cnt_loop) + '-----')                   
                cloud=current_cloud[idx:idx+batch_size]
                label=current_label[idx:idx+batch_size]
            
                idx+=batch_size

                #feeding 
                feed_dict = {ops['pointclouds_pl']: cloud,     #(B,N,c)           
                                ops['labels_pl']: label,      #(B) or (B,N)
                                ops['is_training_pl']: is_training,}

                #run the session and get all the results we need
                summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], 
                    ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

                #then cook
                '''
                train_writer.add_summary(summary, step)
                '''
                pred_val = np.argmax(pred_val, 1)   #(B,class_num)->(B,1)                              #supervised&hard learning
                correct = np.sum(pred_val == label[start_idx:end_idx])                         #a bool tensor,sum it then get a number represents the correct_predicted results
                total_correct += correct
                total_seen += BATCH_SIZE
                loss_sum += loss_val
        
       
    log_string('mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
"""finally we have mean-loss & aaccuracy from training b*len(TRAIN_FILES) sets of data"""

#************************************************test unit*******************************************************    
def testOReval_one_epoch(sess, ops,ops_data,testOReval_writer,file_lenth,batch_size=BATCH_SIZE):


    is_training = False   
    loss_sum = 0.      
    total_seen = 0  
    total_correct=0.  
    total_seen_class = [0 for _ in range(CLASS_NUM)]
    total_correct_class = [0 for _ in range(CLASS_NUM)]


    cnt_loop=0
    for i in range(file_lenth):  #tfrecords   

        with tf.variable_scope('data_packer'):      
            list_cloud,list_label=sess.run([ops_data['list_cloud_%s'%(i)],ops_data['list_label_%s'%(i)]])
            current_cloud,current_label=tl.data_loader(list_cloud,list_label)
    
        idx=0
        num=int(current_cloud.shape[0])
        
        with tf.variable_scope('test/vali_layer') as sc:    
            while idx<num:              
                log_string('----' + 'test/vali-batch-count='+str(cnt_loop) + '-----')                   
                cloud=current_cloud[idx:idx+batch_size]
                label=current_label[idx:idx+batch_size]
                #labels=np.squeeze(labels)
                idx+=batch_size
        
                 #feeding 
                feed_dict = {ops['pointclouds_pl']: cloud,     #(B,N,c)           
                                ops['labels_pl']: label,      #(B) or (B,N)
                                ops['is_training_pl']: is_training,}


                summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],             #(pred_val:(B,CLASS_NUM))
                    ops['loss'], ops['pred']], feed_dict=feed_dict)

                testOReval_writer.add_summary(summary, step)
                                   
                print('loss(test):',loss_val)

                #then cook  
                pred_val = np.argmax(pred_val, 1)    
                correct = np.sum(pred_val ==label)  #in this line ,labels have been changed to float32
                total_correct += correct
                total_seen += BATCH_SIZE
                loss_sum += (loss_val*BATCH_SIZE)
                #extra loop:
                #count how many times each class has occured in prediction,and numbering the correct ones
                templ=np.squeeze(label[:,-1])   #(B)               
                for i in range(batch_size):  
                    l = int(templ [i])    #(1)   
                    total_correct_class[l] += (pred_val[i] == l)   
                    total_seen_class[l] += 1    
               
                cnt_loop+=1
                       
    log_string('test mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('test accuracy: %f'% (total_correct / float(total_seen)))
    log_string('test mean class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
         


if __name__ == "__main__":

    train()
    LOG_FOUT.close()