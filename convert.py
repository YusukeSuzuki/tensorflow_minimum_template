from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf

import simple_autoencoder.model as model
import simple_autoencoder.utils as utils

DEFAULT_MINIBATCH_SIZE=1
DEFAULT_INPUT_MODEL='model_training'
DEFAULT_OUTPUT_MODEL='model'

DEFAULT_INPUT_THREADS=8
DEFAULT_INPUT_QUEUE_MIN=2000
DEFAULT_INPUT_QUEUE_MAX=10000

DEFAULT_INPUT_WIDTH=128
DEFAULT_INPUT_HEIGHT=128

def create_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('-m','--minibatch-size', type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument('-i','--input-model', type=str, default=DEFAULT_INPUT_MODEL)
    parser.add_argument('-o','--output-model', type=str, default=DEFAULT_OUTPUT_MODEL)

    parser.add_argument('--input_threads', type=int, default=DEFAULT_INPUT_THREADS)
    parser.add_argument('--input_queue_min', type=int, default=DEFAULT_INPUT_QUEUE_MIN)
    parser.add_argument('--input_queue_max', type=int, default=DEFAULT_INPUT_QUEUE_MAX)
    return parser

def add_application_arguments(parser):
    parser.add_argument('--width', type=int, default=DEFAULT_INPUT_WIDTH)
    parser.add_argument('--height', type=int, default=DEFAULT_INPUT_HEIGHT)
    return parser

def main():
    parser = create_argument_parser()
    parser = add_application_arguments(parser)
    args = parser.parse_args()
    proc(args)

def proc(args):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.name_scope('input'):
            images = tf.placeholder(name='images_placeholder',
                shape=[args.minibatch_size, args.height, args.width, 3], dtype=tf.float32)
            tf.add_to_collection('images_placeholder',images)

        with tf.device('/gpu:0'):
            with tf.variable_scope('model'):
                out_images = model.inference(images)
                out_images = tf.identity(out_images, name='out_node')
                tf.add_to_collection('inference_op',out_images)

        inference_variables = [
            v for v in tf.trainable_variables() if v.name.count('/inference/')]

        # create session
        config_proto = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session( config=config_proto)

        # ready to run
        sess.run(tf.global_variables_initializer())

        inference_saver = tf.train.Saver(var_list=inference_variables)
        #inference_saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(args.input_model)

        if latest_checkpoint:
            inference_saver.restore(sess, latest_checkpoint)
        else:
            raise EnvironmentError('no checkpoint file')

        # run
        converted_dir = Path(args.output_model)
        converted_dir.mkdir(parents=True, exist_ok=True)
        tf.train.write_graph(sess.graph.as_graph_def(), str(converted_dir), 'model.pb')
        inference_saver.save(sess, str(converted_dir/'model'))

if __name__ == '__main__':
    main()


