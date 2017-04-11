#!/bin/bash

tfpath=`pip show tensorflow | grep "Location: \(.\+\)$" | sed 's/Location: //'`
python $tfpath/tensorflow/python/tools/freeze_graph.py --input_checkpoint model/model --output_node_names=model/out_node --output_graph=graph.pb --input_graph=model/model.pb

