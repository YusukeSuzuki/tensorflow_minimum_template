健康で文化的な最低限度のTensorFlowコマンドラインアプリのテンプレート
------------------------------------------------------------------------------------------------

このリポジトリは以下の流れでTensorFlowを用いた単体のコマンドラインツールを作るためのテンプレートです。

- モデルを作る
- 学習する
- オプティマイザなどを取り除いてモデルファイルを小さくする
- モデルをcheckpointファイルからProtocolBuffersファイルに変換する
- アプリケーションスクリプトで利用する

ディレクトリ構成は以下のようになります。

```text
../tensorflow_minimum_template/
|-- LICENSE
|-- README.md
|-- convert.py
|-- execute.py
|-- make_graph_pb.sh
|-- simple_autoencoder
|   |-- __init__.py
|   |-- model.py
|   `-- utils.py
`-- train.py
```

model.pyにモデルが記述されます。説明のためのシンプルなコンボリューショナルオートエンコーダです。実際のアプリケーションではもっと凝ったことをすればよいでしょう。
train.pyは学習のためのコマンドです。ラーニングレートなどはここで調整します。このテンプレートではGPUを1つ使用します。CPUしかない場合は適宜書き換えます。
convert.pyは学習したモデルのチェックポイントファイルから推論部文だけを取り出して別のチェックポイントファイルに書き出します。
make_graph_pb.shでチェックポイントファイルをProtocolBuffersファイルに変換します。スクリプトの中でTensorFlow付属のfreeze_graph.pyを使用しています。
execute.pyがコマンドラインアプリケーションです。画像ファイルを読み込んでオートエンコーダを通して出力を保存します。

使い方
-------------------------------------------------

### 1. モデルを学習します

学習のセッションをそのまま保存するとファイルサイズはかなり大きくなります。

```bash
$ python train.py --samplesdir-/path/to/sample/jpg/directory --num-iterations-3000
$ ls -lh model_training/
total 29M
-rw-rw-r-- 1 user user  115 Apr 11 00:58 checkpoint
-rw-rw-r-- 1 user user 1.3M Apr 11 00:57 model-0.data-00000-of-00001
-rw-rw-r-- 1 user user  788 Apr 11 00:57 model-0.index
-rw-rw-r-- 1 user user  14M Apr 11 00:57 model-0.meta
-rw-rw-r-- 1 user user 1.3M Apr 11 00:58 model-1000.data-00000-of-00001
-rw-rw-r-- 1 user user  788 Apr 11 00:58 model-1000.index
-rw-rw-r-- 1 user user  14M Apr 11 00:58 model-1000.meta
```

### 2. 学習結果のチェックポイントファイルを小さくする

学習した結果のチェックポイントファイルからオプティマイザの情報等を取り除いて小さくします。

```bash
$ python convert.py
$ ls -lh model/
total 448K
-rw-rw-r-- 1 user user   67 Apr 11 01:00 checkpoint
-rw-rw-r-- 1 user user 419K Apr 11 01:00 model.data-00000-of-00001
-rw-rw-r-- 1 user user  311 Apr 11 01:00 model.index
-rw-rw-r-- 1 user user  18K Apr 11 01:00 model.meta
```

### 3. チェックポイントファイルをProtocolBuffersファイルに変換する

チェックポイントファイルではグラフを読み込むのに複数のファイルのセットが必要でしたがProtocolBuffersに変換すればひとつにまとめられます。

```bash
$ bash make_graph_pb.sh
$ ls -lh graph.pb
-rw-rw-r-- 1 user user 422K Apr 12 00:40 graph.pb
```

### 3. モデルをアプリケーションで使用する

ProtocolBuffersファイルを読み込んでグラフを作成し画像ファイルに使用します。

```bash
$ python execute.py /path/to/your/image/jpg/file.jpg
$ your_favorite_image_viewer out.jpg
```

