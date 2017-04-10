import tensorflow as tf
import math

IMAGE_SUMMARY_MAX_OUTPUTS = 3

def inference(images):
    # format = 'NHWC'
    kernel_size = 5
    out = images

    with tf.variable_scope('inference'):
        with tf.variable_scope('conv1'):
            prev_shape = out.get_shape().as_list()
            stddev = math.sqrt(2 / (kernel_size * kernel_size*prev_shape[3]))
            w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev),
                shape=[kernel_size, kernel_size, prev_shape[3], 32], name='weight')
            out = tf.nn.conv2d(out, w, padding='SAME', strides=[1,1,1,1])
            out = tf.nn.relu(out)
        with tf.variable_scope('downscale'):
            prev_shape = out.get_shape().as_list()
            stddev = math.sqrt(2 / (kernel_size * kernel_size*prev_shape[3]))
            w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev),
                shape=[kernel_size, kernel_size, prev_shape[3], 64], name='weight')
            out = tf.nn.conv2d(out, w, padding='SAME', strides=[1,2,2,1])
            out = tf.nn.relu(out)
        with tf.variable_scope('upscale'):
            reverse_shape = prev_shape
            prev_shape = out.get_shape().as_list()
            stddev = math.sqrt(2 / (kernel_size * kernel_size*prev_shape[3]))
            w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev),
                shape=[kernel_size, kernel_size, 32, prev_shape[3]], name='weight')
            out = tf.nn.conv2d_transpose(out, w, padding='SAME', strides=[1,2,2,1],
                output_shape=reverse_shape)
            out = tf.nn.relu(out)
        with tf.variable_scope('output'):
            prev_shape = out.get_shape().as_list()
            stddev = math.sqrt(2 / (kernel_size * kernel_size*prev_shape[3]))
            w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev),
                shape=[kernel_size, kernel_size, prev_shape[3], 3], name='weight')
            out = tf.nn.conv2d(out, w, padding='SAME', strides=[1,1,1,1])
            out = tf.nn.relu(out)

        tf.summary.image('inference', out, max_outputs=IMAGE_SUMMARY_MAX_OUTPUTS,
            collections=[tf.GraphKeys.SUMMARIES, 'inference', 'image'])

    return out

def loss(label, inference):
    with tf.name_scope('loss'):
        out = tf.squared_difference(label, inference)
        out_mean = tf.reduce_mean(out)
        tf.summary.scalar(
            'loss', out_mean, collections=[tf.GraphKeys.SUMMARIES, 'loss', 'scalar'])
    return out

def train(loss, global_step):
    with tf.name_scope('train'):
        opt = tf.train.AdamOptimizer(1e-5)
        grads = opt.compute_gradients(loss)
        out = opt.apply_gradients(grads, global_step=global_step)

    return out

