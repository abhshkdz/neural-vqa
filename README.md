# neural-vqa

This is an experimental Torch implementation of the
VIS + LSTM visual question answering model from the paper
[Exploring Models and Data for Image Question Answering][2]
by Mengye Ren, Ryan Kiros & Richard Zemel.

## Setup

Requirements:

- [Torch][10]
- [loadcaffe][9]

Download the MSCOCO train+val images and [VQA][1] data using `sh data/download_data.sh`.
If you have them downloaded, copy over the `train2014` and `val2014` image folders
and VQA JSON files to the `data` folder.

Download the [VGG-19][7] Caffe model and prototxt using `sh models/download_models.sh`.

### Known issues

- To avoid memory issues with LuaJIT, install Torch with vanilla Lua.
More instructions [here][4].
- If working with plain Lua, [luaffifb][8] may be needed for [loadcaffe][9],
unless using pre-extracted fc7 features.

## Usage

### Extract image features

```
th extract_fc7.lua -split train
th extract_fc7.lua -split val
```

#### Options

- `batch_size`: Batch size. Default is 10.
- `split`: train/val. Default is `train`.
- `gpuid`: 0-indexed id of GPU to use. Default is -1 = CPU.
- `proto_file`: Path to the `deploy.prototxt` file for the VGG Caffe model. Default is `models/VGG_ILSVRC_19_layers_deploy.prototxt`.
- `model_file`: Path to the `.caffemodel` file for the VGG Caffe model. Default is `models/VGG_ILSVRC_19_layers.caffemodel`.
- `data_dir`: Data directory. Default is `data`.
- `feat_layer`: Layer to extract features from. Default is `fc7`.
- `input_image_dir`: Image directory. Default is `data`.


### Training

```
th train.lua
```

#### Options

- `rnn_size`: Size of LSTM internal state. Default is 1024.
- `embedding_size`: Size of word embeddings. Default is 200.
- `learning_rate`: Learning rate. Default is 5e-4.
- `learning_rate_decay`: Learning rate decay factor. Default is 0.95.
- `learning_rate_decay_after`: In number of epochs, when to start decaying the learning rate. Default is 10.
- `decay_rate`: Decay rate for RMSProp. Default is 0.95.
- `batch_size`: Batch size. Default is 64.
- `max_epochs`: Number of full passes through the training data. Default is 50.
- `dropout`:  Dropout for regularization. Probability of dropping input. Default is 0.5.
- `init_from`: Initialize network parameters from checkpoint at this path.
- `save_every`: No. of iterations after which to checkpoint. Default is 1000.
- `train_fc7_file`: Path to fc7 features of training set. Default is `data/train_fc7.t7`.
- `fc7_image_id_file`: Path to fc7 image ids of training set. Default is `data/train_fc7_image_id.t7`.
- `val_fc7_file`: Path to fc7 features of validation set. Default is `data/val_fc7.t7`.
- `val_fc7_image_id_file`: Path to fc7 image ids of validation set. Default is `data/val_fc7_image_id.t7`.
- `data_dir`: Data directory. Default is `data`.
- `checkpoint_dir`: Checkpoint directory. Default is `checkpoints`.
- `savefile`: Filename to save checkpoint to. Default is `vqa`.
- `gpuid`: 0-indexed id of GPU to use. Default is -1 = CPU.


## Implementation Details

- Last hidden layer image features from [VGG-19][6]
- [GloVe][5] 200d word embeddings as question features
- Zero-padded question sequences for batched implementation
- Training questions are filtered for `top_n` answers,
`top_n = 1000` by default (~87% coverage)

[1]: http://visualqa.org/
[2]: http://arxiv.org/abs/1505.02074
[3]: http://arxiv.org/abs/1505.00468
[4]: https://github.com/torch/distro
[5]: http://nlp.stanford.edu/projects/glove/
[6]: http://arxiv.org/abs/1409.1556
[7]: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
[8]: https://github.com/facebook/luaffifb
[9]: https://github.com/szagoruyko/loadcaffe
[10]: http://torch.ch/
