# おおまかな構造はtrain.pyと同じです
from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf

import simple_autoencoder.model as model
import simple_autoencoder.utils as utils

# コマンドラインアプリでは1回1枚の画像を変換することにしてミニバッチサイズを1にします
DEFAULT_MINIBATCH_SIZE=1
DEFAULT_INPUT_MODEL='model_training'
DEFAULT_OUTPUT_MODEL='model'

DEFAULT_INPUT_WIDTH=128
DEFAULT_INPUT_HEIGHT=128

def create_argument_parser():
    # 学習はしないのでインプットキュー関係の引数はなくなります
    parser = ArgumentParser()
    parser.add_argument('-m','--minibatch-size', type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument('-i','--input-model', type=str, default=DEFAULT_INPUT_MODEL)
    parser.add_argument('-o','--output-model', type=str, default=DEFAULT_OUTPUT_MODEL)

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
    # 学習した変数の値を使いつつコマンドライン用の別のグラフを作ります。

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # 学習時はファイルリストから画像を読み込みtf.train.shuffle_batchで
        # モデルに渡していました。
        # アプリでの利用のためにtf.placeholderでの画像渡しに変えておきます。
        with tf.name_scope('input'):
            images = tf.placeholder(name='images_placeholder',
                shape=[args.minibatch_size, args.height, args.width, 3], dtype=tf.float32)
            tf.add_to_collection('images_placeholder',images)

        # 推論モデルの作成
        # CPUのみを使用したい場合はここを書き換えてください。
        # 本来ならコマンドラインオプションにするところ。
        with tf.device('/gpu:0'):
            with tf.variable_scope('model'):
                # 推論のみなのでロスや学習のオペレーションは作りません
                out_images = model.inference(images)
                # アプリで使用するため推論に名前をつけておきます
                out_images = tf.identity(out_images, name='out_node')
                tf.add_to_collection('inference_op',out_images)

        # 推論に関わる変数のみを書き出すために名前で変数を集めておきます。
        # もしかしたら必要ないかも知れない。
        inference_variables = [
            v for v in tf.trainable_variables() if v.name.count('/inference/')]

        # セッション作成
        config_proto = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session( config=config_proto)

        # 変数を初期化
        sess.run(tf.global_variables_initializer())

        # 学習したチェックポイントファイルから変数を復元します。
        inference_saver = tf.train.Saver(var_list=inference_variables)
        latest_checkpoint = tf.train.latest_checkpoint(args.input_model)

        if latest_checkpoint:
            inference_saver.restore(sess, latest_checkpoint)
        else:
            raise EnvironmentError('no checkpoint file')

        # あらためて推論のみのグラフをチェックポイントファイルとして書き出します。
        # graph_defファイルも書き出しておきます。
        converted_dir = Path(args.output_model)
        converted_dir.mkdir(parents=True, exist_ok=True)
        tf.train.write_graph(sess.graph.as_graph_def(), str(converted_dir), 'model.pb')
        inference_saver.save(sess, str(converted_dir/'model'))

if __name__ == '__main__':
    main()

