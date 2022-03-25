import tensorflow as tf
import numpy as np 
from model import config
layers=tf.layers

def leaky_relu(alpha):
    
    def op(inputs):
        return tf.nn.leaky_relu(inputs,alpha=alpha)
    return op


def maxout(inputs,num_units, axis=-1, scope=None):
    with tf.variable_scope(scope, 'MaxOut', [inputs]):
    
        shape = inputs.get_shape().as_list()
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('number of features({}) is not '
                            'a multiple of num_units({})'.format(
                                num_channels, num_units))
        shape[axis] = num_units
        shape += [num_channels // num_units]

        # Dealing with batches with arbitrary sizes
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = tf.shape(inputs)[i]
        
        outputs = tf.reduce_max(
           tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs

def conv_block(inputs,filters,kernel_size,strides,name,padding='valid',
               activation=leaky_relu(config.alpha),
               kernel_initializer=tf.contrib.layers.xavier_initializer()):
    net=layers.conv2d(inputs,filters=filters,kernel_size=kernel_size,
                      activation=activation,
                      kernel_initializer=kernel_initializer,
                      strides=strides,name=name)
    return net

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  
    if stride == 1:
        return layers.conv2d(
            inputs,
            filters = num_outputs,
            kernel_size = kernel_size,
            strides=1,
            dilation_rate=rate,
            padding='same',
            name=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(
            inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return layers.conv2d(
            inputs,
            num_outputs,
            kernel_size,
            strides=stride,
            dilation_rate=rate,
            padding='valid',
            name=scope)

def subsample(inputs,factor,scope=None):
    '''
    通过pooling将inputs的shape调整成最后卷积处理后的大小
    args:
    inputs: a tensor [batch,h,w,c]
    kernel_size:
    factor: stride
    return subsample output
    '''
    if  factor == 1:
        return inputs
    else:
        return layers.max_pooling2d(inputs,pool_size=1, strides = factor,name=scope)


       
def bottleneck(inputs,depth,depth_bottleneck,kernel_size,stride,scope,is_training=True,padding='same',
               activation_fn=leaky_relu(config.alpha),
               kernel_initializer=tf.contrib.layers.xavier_initializer()):
    '''
    residual block

    args:
    inputs: input tensor  [batch,h,w,c]
    depth: final output depth
    depth_bottleneck: middel output depth

    return: output tensor
    '''

    depth_in=inputs.get_shape().as_list()[-1]
    with tf.variable_scope(scope) :
        
        preact=layers.batch_normalization(inputs, training=is_training)
        preact= activation_fn(preact)
        
        if depth == depth_in:
            shortcut = subsample(inputs,stride,'shortcut')
        
        else:
          
            shortcut  = layers.conv2d(preact,filters=depth,kernel_size=1,
                        activation=None,padding=padding,
                        kernel_initializer=kernel_initializer,
                        strides=stride,name='shortcut')
        
        residual = layers.conv2d(preact,filters=depth_bottleneck,kernel_size=1,
                        padding=padding,
                        activation=activation_fn,
                        kernel_initializer=kernel_initializer,
                        strides=1,name='conv1')
        
        residual = layers.conv2d(residual,filters=depth_bottleneck,kernel_size=kernel_size,
                        padding=padding,
                        activation=activation_fn,
                        kernel_initializer=kernel_initializer,
                        strides=stride,name='conv2')
        
        residual = layers.conv2d(residual ,filters=depth,kernel_size=1,
                        activation=None,
                        padding=padding,
                        kernel_initializer=kernel_initializer,
                        strides=1,name='conv3')
        
        
        output = shortcut + residual
        
    return output




def build_network(images,is_training):
    #images:Bx64x64x3

    net = conv2d_same(images,16,3,2,scope='convd_same1')#Bx32x32x16

    net=bottleneck(net,16*4,16,3,2,scope='block1',is_training=is_training)#BX16X16X64
    net=bottleneck(net,16*4,16,3,2,scope='block2',is_training=is_training)#BX8X8X64
    net=bottleneck(net,32*4,32,3,1,scope='block3',is_training=is_training)#BX8X8X128
    net=bottleneck(net,32*4,32,3,1,scope='block4',is_training=is_training)#BX8X8X128
    net=bottleneck(net,16*4,16,3,2,scope='block5',is_training=is_training)#BX4X4X64
    net=bottleneck(net,16*4,16,3,2,scope='block6',is_training=is_training)#BX4X4X64
    net=layers.batch_normalization(net, training=is_training)
    net= leaky_relu(config.alpha)(net)
    net=layers.flatten(net,name='flat')
    net=layers.dense(net,256,activation=leaky_relu(config.alpha),name='fc1')#BX256
    net=layers.dense(net,config.num_class,activation=None,name='logits')#BXconfig.num_class
    
    return net
  


def compute_loss(images,labels,phase):
    
    logits=build_network(images,phase)
    with tf.variable_scope('loss'):
        class_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='class_loss'))
        train_var_list=[v for v in tf.trainable_variables()
                    if 'beta' not in v.name and 'gamma' not in v.name]
        l2_loss=config.weight_decay*tf.add_n(
            [tf.nn.l2_loss(v) for v in train_var_list])
        total_loss=class_loss+l2_loss
        
        correct_prediction = tf.equal(tf.argmax(logits,1),tf.cast(labels,dtype=tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
       
        return total_loss,class_loss,accuracy
 


