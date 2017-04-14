#!/bin/bash

# tensorflowのインストールパスを取得します。
# pipでインストールしていることを想定しています。
# ネイティブでもpyenvでも取得できるはず。多分。
# 使用しているTensorFlowのバージョンと一致したfreeze_graph.pyが必要なためです。
tfpath=`pip show tensorflow | grep "Location: \(.\+\)$" | sed 's/Location: //'`

# tensorflowパッケージに付属しているfreeze_graph.pyを使用してチェックポイントファイルから
# graph_def ProtocolBuffersファイルに変換します。
python $tfpath/tensorflow/python/tools/freeze_graph.py \
    --input_checkpoint model/model --output_node_names=model/out_node \
    --output_graph=graph.pb --input_graph=model/model.pb

