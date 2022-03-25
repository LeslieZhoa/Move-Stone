'''
将tensorflow的ckpt模型文件转换成.pb文件并压缩
'''
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
# import networks  
from tensorflow.python.tools import freeze_graph 
from tensorflow.python.platform import gfile

import utils.config as Config
from utils.model import *

 
def freeze_model():
  with tf.Graph().as_default():
    # Create input and target placeholder.
    images=tf.placeholder(tf.float32,[None,Config.image_size,Config.image_size,3],name='input')
    phase=tf.placeholder(tf.bool,name='phase')
    logits=build_network(images,phase)
    tf.identity(logits,name='output')
    
 
    
    # Create a saver and load.
    saver = tf.train.Saver()
     
   
    config = tf.ConfigProto(allow_soft_placement=True)
   
    sess = tf.Session(config=config)
 
    pretrained_model_checkpoint_path = r"D:/code/test/handTest/r4/model/hand_gesture"
 
    # Restore checkpoint from file.
    if pretrained_model_checkpoint_path:
      assert tf.gfile.Exists(pretrained_model_checkpoint_path)
      ckpt = tf.train.get_checkpoint_state(
               pretrained_model_checkpoint_path)
      restorer = tf.train.Saver()
      restorer.restore(sess, ckpt.model_checkpoint_path)
      print('%s: Pre-trained model restored from %s' %
        (datetime.now(), ckpt.model_checkpoint_path))
 
    tf.train.write_graph(sess.graph_def, 'D:/code/test/handTest/r4/model/output_model/pb_model', 'model.pb')
    
    freeze_graph.freeze_graph('D:/code/test/handTest/r4/model/output_model/pb_model/model.pb', '', False, ckpt.model_checkpoint_path, 'output,phase','save/restore_all', 'save/Const:0', 'D:/code/test/handTest/r4/model/output_model/pb_model/frozen_model.pb', False, "")
    #..............................................................................................................................输出节点名可以多个用，隔开 
 
 
if __name__ == '__main__':
 
  # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
  freeze_model()

 
def read():
  '''
  读取.pb文件
  '''
   with tf.Graph().as_default():
           output_graph_def = tf.GraphDef()
           output_graph_path = "D:/code/test/handTest/r4/model/output_model/pb_model/frozen_model.pb"

       with open(output_graph_path, 'rb') as f:
           output_graph_def.ParseFromString(f.read())
           _ = tf.import_graph_def(output_graph_def, name="")

       with tf.Session() as sess:
           sess.run(tf.global_variables_initializer())
           graph=sess.graph
           images= sess.graph.get_tensor_by_name("input:0")

           phase = sess.graph.get_tensor_by_name("phase:0")
           logits = sess.graph.get_tensor_by_name("output:0")
