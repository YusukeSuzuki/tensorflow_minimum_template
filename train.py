# TensorFlowだと独自のコマンドライン引数パーサを使用していますがここでは
# python3標準のArgumentParserを使用します。
# TensorFlowが独自のものを使用するのはおそらくPython 2/3両対応のためでしょう。
from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf

# 自分のモデルをインポートしておきます
import simple_autoencoder.model as model
import simple_autoencoder.utils as utils

# 学習におけるイテレーション数など各種パラメータのデフォルト値です。
# 基本的にすべてコマンドライン引数で上書きできるようにしておきます。
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
    # コマンドライン引数のパーサを作ります。
    # まず大体のアプリケーションで共通して使う引数を設定します。
    parser = ArgumentParser()
    parser.add_argument('-n','--num-iterations', type=int, default=DEFAULT_NUM_ITERATIONS)
    parser.add_argument('-m','--minibatch-size', type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument('-l','--logdir', type=str, default=DEFAULT_LOGDIR)
    parser.add_argument('-s','--samplesdir', type=str, default=DEFAULT_SAMPLESDIR)
    parser.add_argument('-i','--input-model', type=str, default=DEFAULT_INPUT_MODEL)
    parser.add_argument('-o','--output-model', type=str, default=DEFAULT_OUTPUT_MODEL)


    parser.add_argument('--input-threads', type=int, default=DEFAULT_INPUT_THREADS)
    parser.add_argument('--input-queue_min', type=int, default=DEFAULT_INPUT_QUEUE_MIN)
    parser.add_argument('--input-queue-max', type=int, default=DEFAULT_INPUT_QUEUE_MAX)
    return parser

def add_application_arguments(parser):
    # 今回のアプリケーション（オートエンコーダ）固有の引数を追加します。
    # ここらへんの切り分けはほとんど趣味です。
    parser.add_argument('--width', type=int, default=DEFAULT_INPUT_WIDTH)
    parser.add_argument('--height', type=int, default=DEFAULT_INPUT_HEIGHT)
    return parser

def main():
    # メインです。
    # コマンドライン引数のパーサを作ってパースして本処理に投げるだけ
    parser = create_argument_parser()
    parser = add_application_arguments(parser)
    args = parser.parse_args()
    proc(args)

def proc(args):
    # 学習サンプルの画像ファイルのリストを取得します。
    # まずはディレクトリを開く
    sample_dir = Path(args.samplesdir)

    # ディレクトリの直下からjpgファイルのパスを取得します。
    # Path.globを使用すればもうちょっと凝ったことができます。
    file_list = [p for p in sample_dir.iterdir() if p.suffix == '.jpg']
    file_list = list(map(str, file_list))

    # 学習のグラフを作っていきます。
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # まずはグローバルステップ。
        # 複数回に分けて学習するときには独自に作っておくと便利です。
        with tf.variable_scope('global'):
            global_step = tf.get_variable(
                'global_step', shape=[], initializer=tf.constant_initializer(0, dtype=tf.int64))

        # 次に画像ファイルの読み込みキューを作ります。
        # ここらへんはチュートリアルでもよく出てきます。
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(file_list)
            reader = tf.WholeFileReader()
            images = tf.train.shuffle_batch(
                [utils.read_image_op(filename_queue, reader, args.height, args.width)],
                args.minibatch_size, args.input_queue_max, args.input_queue_min,
                num_threads=args.input_threads)

        # 推論と学習を組んでいきます。
        # GPUを1つ使用します。CPUしかない場合はここを書き換えることになります。
        # ここもコマンドライン引数にしてしまってもよいでしょう。
        with tf.device('/gpu:0'):
            with tf.variable_scope('model'):
                # 推論
                out_images = model.inference(images)
                # あとで使用するために推論に別名をつけておきます
                out_images = tf.identity(out_images, name='out_node')
                # ロスの計算
                loss = model.loss(images, out_images)
                # 学習オペレーションの取得
                train_op = model.train(loss, global_step)

        # ログ出力オペレーションを取得しておきます。
        log_op = tf.summary.merge_all()

        # セッションを作成します。
        # allow_soft_placement=True しておけばGPUでsummaryオペレーションを作っても
        # 善きに計らってくれます。
        # XLAなどの最適化もここで指定しますが割愛。
        config_proto = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session( config=config_proto)

        # ログ書き出しオブジェクトを作ります。
        # 実際にはもう少し詳しくディレクトリを分けたほうがよいでしょう
        # 例: ./logs/train, ./logs/test など
        writer = tf.summary.FileWriter('./logs')

        # tf.Variable を初期化します。
        sess.run(tf.global_variables_initializer())

        # チェックポイントファイルのIOオブジェクトを作ります。
        saver = tf.train.Saver()

        # モデル出力先ディレクトリを作成します。
        training_dir = Path(args.output_model)
        training_dir.mkdir(parents=True, exist_ok=True)

        # 以前に学習したモデルが存在すればそれをレストアして続きの学習とします。
        latest_checkpoint = tf.train.latest_checkpoint(str(args.input_model))

        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)

        # ログにグラフ構造を書き出しておきます。
        # こうしておくことでTensorBoardのGraphタブからグラフの構造を確認できます。
        writer.add_graph(tf.get_default_graph())

        # 入力にキューを使用しているのでスレッドをスタートします。
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # 指定イテレーション分学習を行います
        for i in range(args.num_iterations):
            # 既定グローバルステップごとにモデルを書き出してバックアップとします
            gs = tf.train.global_step(sess, global_step)
            if gs % 5000 == 0:
                saver.save(sess, str(training_dir/'model'), global_step=gs)

            # 10回ごとにロスなどのログを取得して書き出します。
            # scalarのログであればそれほど重くはありませんが画像やヒストグラムは
            # 書き出し処理が重くなりがちなので回数は適宜調整したほうがよいでしょう。
            # モデルや規模によりますが勾配のヒストグラムは500回から1000回ごとで十分だと思います。
            if gs % 10 == 0:
                print("global_step = {}".format(gs))
                _, logs =  sess.run([train_op, log_op])
                writer.add_summary(logs, i)
            else:
                _ = sess.run([train_op])

        # イテレーションが終了したら完成モデルを書き出します。
        gs = tf.train.global_step(sess, global_step)
        saver.save(sess, str(training_dir/'model'), global_step=gs)

        # 開放処理はプロセスにまかせます

if __name__ == '__main__':
    main()

