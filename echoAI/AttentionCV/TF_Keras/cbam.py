import tensorflow as tf

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)
 
 
def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)
 
def conv2d(input, filter, strides, padding="SAME", name=None):
    # filters with shape [filter_height * filter_width * in_channels, output_channels]
    # Must have strides[0] = strides[3] =1
    # For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]
    '''
    Args:
        input: A Tensor. Must be one of the following types: float32, float64.
        filter: A Tensor. Must have the same type as input.
        strides: A list of ints. 1-D of length 4. The stride of the sliding window for each dimension of input.
        padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
        use_cudnn_on_gpu: An optional bool. Defaults to True.
        name: A name for the operation (optional).
    '''
    return tf.nn.conv2d(input, filter, strides, padding=padding, name=name)  # padding="SAME"用零填充边界

def channel_attention_module(Input_feature, r):
    # image tensor shape: [batch, height, width, channel]
    with tf.name_scope('Channel_Attention_Module'):
        tensor_shape = Input_feature.get_shape().as_list()
        height = tensor_shape[-3]
        width = tensor_shape[-2]
        channel = tensor_shape[-1]
        with tf.name_scope('F_max'):
            F_max = tf.nn.max_pool(Input_feature, ksize = [1,height,width,1], strides = [1,1,1,1], padding = 'VALID')
            F_max = tf.reshape(F_max, [-1, channel])
        with tf.name_scope('F_avg'):
            F_avg = tf.nn.avg_pool(Input_feature, ksize = [1,height,width,1], strides = [1,1,1,1], padding = 'VALID')
            F_avg = tf.reshape(F_avg, [-1, channel])
        with tf.name_scope('Shared_MLP'):
            with tf.name_scope('Variable'):
                W0 = weight_variable([channel, int(channel/r)], name='W0')
                W1 = weight_variable([int(channel/r), channel], name='W1')
            hidden_layer_m = tf.nn.relu(tf.matmul(F_max, W0))
            hidden_layer_a = tf.nn.relu(tf.matmul(F_avg, W0))
            layer_m = tf.matmul(hidden_layer_m, W1)
            layer_a = tf.matmul(hidden_layer_a, W1)
        with tf.name_scope('ADD'):
            channel_attention = layer_m + layer_a
        with tf.name_scope('Sigmoid'):
            channel_attention = tf.nn.sigmoid(channel_attention)
    return tf.reshape(channel_attention, [-1, 1, 1, channel])
    
def spatial_attention_module(channel_refined_feature):
    with tf.name_scope('Spatial_Attention_Module'):
        tensor_shape = channel_refined_feature.get_shape().as_list()
        height = tensor_shape[-3]
        width = tensor_shape[-2]
        channel = tensor_shape[-1]
        with tf.name_scope('Concat'):
            with tf.name_scope('MaxPool'):
                F_max = tf.nn.max_pool(channel_refined_feature, ksize = [1,1,1,channel], strides = [1,1,1,1], padding = 'VALID')
            with tf.name_scope('AvgPool'):
                F_avg = tf.reduce_mean(channel_refined_feature, axis=3)
                F_avg = tf.reshape(F_avg, [-1,height,width,1])
            Fs = tf.concat([F_avg, F_max], 3)
        with tf.name_scope('Conv'):
            with tf.name_scope('Variable'):
                filters = weight_variable([7,7,2,1], name='filter')
                bias = weight_variable([1], name='bias')
            with tf.name_scope("Convolution"):
                layer = conv2d(Fs, filters, strides=[1,1,1,1], padding = 'SAME') + bias
        with tf.name_scope('Sigmoid'):
            spatial_attention = tf.nn.sigmoid(layer)
    return spatial_attention

def CBAM(F, r):
    channel_attention = channel_attention_module(F, r)
    with tf.name_scope('Channel_Refined_Feature'):
        channel_refined_feature = channel_attention * F
    spatial_attention = spatial_attention_module(channel_refined_feature)
    with tf.name_scope('Refined_Feature'):
        refined_feature = spatial_attention*channel_refined_feature
    return refined_feature