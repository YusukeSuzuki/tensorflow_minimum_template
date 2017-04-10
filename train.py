from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf

import simple_autoencoder.model as model
import simple_autoencoder.utils as utils

DEFAULT_NUM_ITERATIONS=100000
DEFAULT_MINIBATCH_SIZE=16
DEFAULT_LOGDIR='./logs'
DEFAULT_SAMPLESDIR='./samples'
DEFAULT_INPUT_MODEL='model_training'
DEFAULT_OUTPUT_MODEL='model_training'

DEFAULT_INPUT_THREADS=8
DEFAULT_INPUT_QUEUE_MIN=2000
DEFAULT_INPUT_QUEUE_MAX=10000

DEFAULT_INPUT_WIDTH=128
DEFAULT_INPUT_HEIGHT=128

def create_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('-n','--num-iterations', type=int, default=DEFAULT_NUM_ITERATIONS)
    parser.add_argument('-m','--minibatch-size', type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument('-l','--logdir', type=str, default=DEFAULT_LOGDIR)
    parser.add_argument('-s','--samplesdir', type=str, default=DEFAULT_SAMPLESDIR)
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
    sample_dir = Path(args.samplesdir)

    file_list = [p for p in sample_dir.iterdir() if p.suffix == '.jpg']
    file_list = list(map(str, file_list))

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.variable_scope('global'):
            global_step = tf.get_variable(
                'global_step', shape=[], initializer=tf.constant_initializer(0, dtype=tf.int64))

        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(file_list)
            reader = tf.WholeFileReader()
            images = tf.train.shuffle_batch(
                [utils.read_image_op(filename_queue, reader, args.height, args.width)],
                args.minibatch_size, args.input_queue_max, args.input_queue_min,
                num_threads=args.input_threads)

        with tf.device('/gpu:0'):
            with tf.variable_scope('model'):
                out_images = model.inference(images)
                loss = model.loss(images, out_images)
                train_op = model.train(loss, global_step)

        log_op = tf.summary.merge_all()

        # create session
        config_proto = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session( config=config_proto)

        # ready to run
        writer = tf.summary.FileWriter('./logs')

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        training_dir = Path(args.output_model)
        training_dir.mkdir(parents=True, exist_ok=True)

        latest_checkpoint = tf.train.latest_checkpoint(str(args.input_model))

        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)

        writer.add_graph(tf.get_default_graph())

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # run

        for i in range(args.num_iterations):
            gs = tf.train.global_step(sess, global_step)
            if gs % 5000 == 0:
                saver.save(sess, str(training_dir/'model'), global_step=gs)

            if gs % 10 == 0:
                print("global_step = {}".format(gs))
                _, logs =  sess.run([train_op, log_op])
                writer.add_summary(logs, i)
            else:
                _ = sess.run([train_op])

        gs = tf.train.global_step(sess, global_step)
        saver.save(sess, str(training_dir/'model'), global_step=gs)

if __name__ == '__main__':
    main()

