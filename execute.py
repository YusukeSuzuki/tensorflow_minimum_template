from argparse import ArgumentParser
from pathlib import Path

# 画像の入出力にpillowを使用します
import tensorflow as tf
import numpy as np
from PIL import Image

DEFAULT_INPUT_MODEL='model'
DEFAULT_INPUT_GRAPHDEF='graph.pb'

DEFAULT_INPUT_WIDTH=128
DEFAULT_INPUT_HEIGHT=128

def create_argument_parser():
    # 学習モデルのグラフを指定します
    # 入出力がテンソルの名前も含めて同一仕様であれば差し替えもできます
    parser = ArgumentParser()
    parser.add_argument('-i','--input-graphdef', type=str, default=DEFAULT_INPUT_GRAPHDEF)
    return parser

def add_application_arguments(parser):
    # 変換する画像の指定
    parser.add_argument('imagefiles', nargs='+', type=str)
    return parser

def main():
    parser = create_argument_parser()
    parser = add_application_arguments(parser)
    args = parser.parse_args()
    proc(args)

def proc(args):
    # セッションを作って
    config_proto = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=config_proto)

    # graph_defファイルを読み込んでデフォルトグラフにします。
    with tf.gfile.FastGFile(args.input_graphdef, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    # 入力のtf.placeholderを取得します
    images_placeholder = tf.get_default_graph().get_tensor_by_name('input/images_placeholder:0')
    # 推論オペレーションを取得します
    inference_op = tf.get_default_graph().get_tensor_by_name('model/out_node:0')

    i = 0

    for imagepath in args.imagefiles:
        # 画像を読み込んでサイズ1のミニバッチの形式にします。
        input_image = Image.open(imagepath)
        input_image = input_image.resize((DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT))
        input_image = np.expand_dims(np.asarray(input_image), axis=0) / 255

        # 入力画像をplaceholderに仕込んで推論オペレーションを実行します。
        out_images = sess.run(
            [inference_op], feed_dict={images_placeholder: input_image})

        # 画像の値を調整してミニバッチから単一画像行列に変換して保存します。
        out_images = np.multiply(out_images, 255)
        out_images = np.squeeze(out_images, axis=(0,1))
        out_image = Image.fromarray(np.uint8(out_images))
        out_image.save('out_{}.jpg'.format(i))
        
        i += 1

if __name__ == '__main__':
    main()

