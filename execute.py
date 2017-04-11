from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image

#import simple_autoencoder.model as model
#import simple_autoencoder.utils as utils

DEFAULT_INPUT_MODEL='model'
DEFAULT_INPUT_GRAPHDEF='graph.pb'

DEFAULT_INPUT_WIDTH=128
DEFAULT_INPUT_HEIGHT=128

def create_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('-i','--input-graphdef', type=str, default=DEFAULT_INPUT_GRAPHDEF)
    return parser

def add_application_arguments(parser):
    parser.add_argument('imagefiles', nargs='+', type=str)
    return parser

def main():
    parser = create_argument_parser()
    parser = add_application_arguments(parser)
    args = parser.parse_args()
    proc(args)

def proc(args):
    config_proto = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=config_proto)

    with tf.gfile.FastGFile(args.input_graphdef, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    images_placeholder = tf.get_default_graph().get_tensor_by_name('input/images_placeholder:0')
    inference_op = tf.get_default_graph().get_tensor_by_name('model/out_node:0')

    for imagepath in args.imagefiles:
        input_image = Image.open(imagepath)
        input_image = input_image.resize((DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT))
        input_image = np.expand_dims(np.asarray(input_image), axis=0) / 255
        out_images = sess.run(
            [inference_op], feed_dict={images_placeholder: input_image})
        out_images = np.multiply(out_images, 255)
        out_images = np.squeeze(out_images, axis=(0,1))
        out_image = Image.fromarray(np.uint8(out_images))
        out_image.save('out.jpg')

if __name__ == '__main__':
    main()

