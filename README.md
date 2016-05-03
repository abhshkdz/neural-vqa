# neural-vqa

[![Join the chat at https://gitter.im/abhshkdz/neural-vqa](https://badges.gitter.im/abhshkdz/neural-vqa.svg)](https://gitter.im/abhshkdz/neural-vqa?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This is an experimental Torch implementation of the
VIS + LSTM visual question answering model from the paper
[Exploring Models and Data for Image Question Answering][2]
by Mengye Ren, Ryan Kiros & Richard Zemel.

![Model architecture](http://i.imgur.com/UXAPlqe.png)

## Setup

Requirements:

- [Torch][10]
- [loadcaffe][9]

Download the [MSCOCO][11] train+val images and [VQA][1] data using `sh data/download_data.sh`. Extract all the downloaded zip files inside the `data` folder.

```
unzip Annotations_Train_mscoco.zip
unzip Questions_Train_mscoco.zip
unzip train2014.zip

unzip Annotations_Val_mscoco.zip
unzip Questions_Val_mscoco.zip
unzip val2014.zip
```

If you had them downloaded already, copy over the `train2014` and `val2014` image folders
and VQA JSON files to the `data` folder.

Download the [VGG-19][7] Caffe model and prototxt using `sh models/download_models.sh`.

### Known issues

- To avoid memory issues with LuaJIT, install Torch with Lua 5.1 (`TORCH_LUA_VERSION=LUA51 ./install.sh`).
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

- `rnn_size`: Size of LSTM internal state. Default is 512.
- `num_layers`: Number of layers in LSTM
- `embedding_size`: Size of word embeddings. Default is 512.
- `learning_rate`: Learning rate. Default is 4e-4.
- `learning_rate_decay`: Learning rate decay factor. Default is 0.95.
- `learning_rate_decay_after`: In number of epochs, when to start decaying the learning rate. Default is 15.
- `alpha`: Alpha for adam. Default is 0.8
- `beta`: Beta used for adam. Default is 0.999.
- `epsilon`: Denominator term for smoothing. Default is 1e-8.
- `batch_size`: Batch size. Default is 64.
- `max_epochs`: Number of full passes through the training data. Default is 15.
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

### Testing

```
th predict.lua -checkpoint_file checkpoints/vqa_epoch23.26_0.4610.t7 -input_image_path data/train2014/COCO_train2014_000000405541.jpg -question 'What is the cat on?'
```

#### Options

- `checkpoint_file`: Path to model checkpoint to initialize network parameters from
- `input_image_path`: Path to input image
- `question`: Question string

## Sample predictions

Randomly sampled image-question pairs from the VQA test set,
and answers predicted by the VIS+LSTM model.

![](http://i.imgur.com/V3nHbo9.jpg)

Q: What animals are those?
A: Sheep

![](http://i.imgur.com/QRBi6qb.jpg)

Q: What color is the frisbee that's upside down?
A: Red

![](http://i.imgur.com/tiOqJfH.jpg)

Q: What is flying in the sky?
A: Kite

![](http://i.imgur.com/4ZmOoUF.jpg)

Q: What color is court?
A: Blue

![](http://i.imgur.com/1D6NxvD.jpg)

Q: What is in the standing person's hands?
A: Bat

![](http://i.imgur.com/tY9BT1I.jpg)

Q: Are they riding horses both the same color?
A: No

![](http://i.imgur.com/hzwj0NS.jpg)

Q: What shape is the plate?
A: Round

![](http://i.imgur.com/n1Kn1vZ.jpg)

Q: Is the man wearing socks?
A: Yes

![](http://i.imgur.com/dXhNKP6.jpg)

Q: What is over the woman's left shoulder?
A: Fork

![](http://i.imgur.com/thzv03r.jpg)

Q: Where are the pink flowers?
A: On wall

## Implementation Details

- Last hidden layer image features from [VGG-19][6]
- Zero-padded question sequences for batched implementation
- Training questions are filtered for `top_n` answers,
`top_n = 1000` by default (~87% coverage)

## Pretrained model and data files

To reproduce results shown on this page or try your own
image-question pairs, download the following and run
`predict.lua` with the appropriate paths.

- vqa\_epoch23.26\_0.4610.t7 (Serialized using Lua51) [[GPU](https://drive.google.com/file/d/0B8qwt8PA_oxpSWhRQ1NKYkxhYnc/view?usp=sharing)] [[CPU](https://drive.google.com/file/d/0B8qwt8PA_oxpbGJQY0EyZ2phYTg/view?usp=sharing)]
- [answers_vocab.t7](https://drive.google.com/file/d/0B8qwt8PA_oxpNE1RdWlMLWlNcVk/view?usp=sharing)
- [questions_vocab.t7](https://drive.google.com/file/d/0B8qwt8PA_oxpd2Y4MXIzb0pxSWM/view?usp=sharing)
- [data.t7](https://drive.google.com/file/d/0B8qwt8PA_oxpejVuTFVsZTJDSUU/view?usp=sharing)

## References

- [Exploring Models and Data for Image Question Answering][2], Ren et al., NIPS15
- [VQA: Visual Question Answering][3], Antol et al., ICCV15

## License

[MIT][12]

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
[11]: http://mscoco.org/
[12]: https://abhshkdz.mit-license.org/
