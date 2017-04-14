# モデルのモジュールです。
# シンプルで学習の結果が見た目にわかりやすいオートエンコーダです。
# このモデル自体はなにかができるわけではありません。
# アプリケーション作成の説明用のものです。
import tensorflow as tf
import math

IMAGE_SUMMARY_MAX_OUTPUTS = 3

def inference(images):
    # 推論部分
    # フォーマットは'NHWC'を想定しています。
    kernel_size = 5
    out = images

    # あとで推論に関わる変数だけ取り出すために全体をスコープでくくります。
    with tf.variable_scope('inference'):
        # あとはレイヤーを重ねるだけ。
        # モデルの解説ではないので詳しいことは省略します。
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

        # ログを出しておきます。
        # コレクションにデフォルト、意味的カテゴリ、型的カテゴリを追加しておくことで
        # ログ出しの切り分けがしやすくなると思います。
        # GPUではログ出しができないのでデバイスを指定すべきですがそれはtf.Sessionの
        # コンフィグで一括で行います。
        tf.summary.image('inference', out, max_outputs=IMAGE_SUMMARY_MAX_OUTPUTS,
            collections=[tf.GraphKeys.SUMMARIES, 'inference', 'image'])

    return out

def loss(label, inference):
    # ロスの定義です。自乗誤差基準です。
    with tf.name_scope('loss'):
        out = tf.squared_difference(label, inference)
        out_mean = tf.reduce_mean(out)
        tf.summary.scalar(
            'loss', out_mean, collections=[tf.GraphKeys.SUMMARIES, 'loss', 'scalar'])
    return out

def train(loss, global_step):
    # 学習の定義です。Adamオプティマイザ
    # 勾配の計算と適用を分割しています。勾配のヒストグラムをログに出せば
    # 学習中に勾配が消失していないか確認できます。
    # ただしログ出力が重くなりがちなので1000イテレーションに1回くらいが良いでしょう。
    # そこまでやらない場合は opt.minimize() で十分です。
    with tf.name_scope('train'):
        opt = tf.train.AdamOptimizer(1e-5)
        grads = opt.compute_gradients(loss)
        out = opt.apply_gradients(grads, global_step=global_step)

        for g, u in grads:
            if g is None:
                continue
            tf.summary.histogram(u.name, g,
                collections=[tf.GraphKeys.SUMMARIES, 'train', 'histogram'])

    return out

