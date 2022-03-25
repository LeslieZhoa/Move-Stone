
import os
import numpy as np
import random

import natsort
import tensorflow as tf
import cv2
import config





class processing:
    def __init__(self,):
       '''
       初始化
       '''
    def read_filename(self,img_path):
        '''
        读取文件返回data
        例：return img,label
        '''
    def tfrecord(self,tfrecord_path,train_pre_list):
        '''
        生成tfrecord
        tfrecord_path:record根目录
        train_pre_list：图片根目录
        '''
        
        

        #make train_tfrecord
        train_tfrecord=os.path.join(tfrecord_path,'train')
        if not os.path.exists(train_tfrecord):
            os.makedirs(train_tfrecord)
        count=len(os.listdir(train_pre_list))#数据总量
        for i,train_img_path in enumerate (os.listdir(train_pre_list)):
            try:
                

                image,label=self.read_filename(os.path.join(train_pre_list, train_img_path))
            
                
            except:
                print('bad image: ',train_img_path)
                continue
            #生成多组record
            if i % 100 == 0 and i>1:
                writer.close()
            if i% 100 == 0:
                writer = tf.python_io.TFRecordWriter(train_tfrecord+'/'+str(i)+".tfrecords")
            
            image=image.tostring()#类型uint8
            label=label.tobytes()#类型float
            #生成record
            example = tf.train.Example(features=tf.train.Features(feature={
                
                'img': _bytes_feature(image),
                'label':_float_feature(label),
                'count':_int64_feature(train_num)}))#int类型

            writer.write(example.SerializeToString())  #序列化为字符串
            print('train: ',i)

        writer.close()
        






def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))




def read_and_decode(filename):
    '''
    读取record
    filename:record的地址，列表形式[]
    '''
    min_after_dequeue = 200
    num_threads = 10 
    capacity = min_after_dequeue + num_threads * config.batch_size
    filename_queue = tf.train.string_input_producer(filename)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(
               serialized_example,
               features={'count': tf.FixedLenFeature([], tf.int64),
                        'img': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.float32) })
    count=tf.reshape(tf.cast(features['count'], tf.int32),[])

    img = tf.decode_raw(features['img'], tf.uint8)
    img=tf.reshape(img,[360,360,3])
    img = tf.cast(img, tf.float32)/256.0
    
    label = tf.reshape(features['label'],[])#自己reshape
    


    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=config.batch_size, capacity=capacity,num_threads=num_threads,
                                                min_after_dequeue=min_after_dequeue)    
  
    
    return img_batch, label_batch,count





