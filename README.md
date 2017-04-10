tensorflow_minimum_template
-------------------------------------------------

This is minimum tensorflow command line application template.
Train model, and convert model file from training checkpoint to inference only checkpoint file for application use.

```text
tensorflow_minimum_template/
|-- LICENSE
|-- README.md
|-- convert.py
|-- execute.py
|-- simple_autoencoder
|   |-- __init__.py
|   |-- model.py
|   `-- utils.py
`-- train.py
```

model.py is code for model. simple convolutional auto-encoder for example.

train.py is training command.
convert.py is model converter.
execute is standalone command line application.

usage
-------------------------------------------------

### 1. train model. it makes big file.

```bash
# python train.py --samplesdir-/path/to/sample/jpg/directory --num-iterations-3000
# ls -lh model_training/
total 29M
-rw-rw-r-- 1 user user  115 Apr 11 00:58 checkpoint
-rw-rw-r-- 1 user user 1.3M Apr 11 00:57 model-0.data-00000-of-00001
-rw-rw-r-- 1 user user  788 Apr 11 00:57 model-0.index
-rw-rw-r-- 1 user user  14M Apr 11 00:57 model-0.meta
-rw-rw-r-- 1 user user 1.3M Apr 11 00:58 model-1000.data-00000-of-00001
-rw-rw-r-- 1 user user  788 Apr 11 00:58 model-1000.index
-rw-rw-r-- 1 user user  14M Apr 11 00:58 model-1000.meta
```

### 2. export inference model file. it remove training data from training model file.

```bash
# python convert.py
# ls -lh model/
total 448K
-rw-rw-r-- 1 user user   67 Apr 11 01:00 checkpoint
-rw-rw-r-- 1 user user 419K Apr 11 01:00 model.data-00000-of-00001
-rw-rw-r-- 1 user user  311 Apr 11 01:00 model.index
-rw-rw-r-- 1 user user  18K Apr 11 01:00 model.meta
```

### 3. use model from command line app

```bash
# python execute.py /path/to/your/image/jpg/file.jpg
# your_favorite_image_viewer out.jpg
```

