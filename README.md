# neural-vqa

This is an experimental Torch implementation of recent models
proposed for visual question answering. Numbers reported on
the [MSCOCO Visual QA dataset][1].

## Results

-- Work in progress --

## Models:

- LSTM+MLP model from the paper [VQA: Visual Question Answering][3] by Antol et al.
- VIS+LSTM model from the paper [Exploring Models and Data for Image Question Answering][2] by Ren et al.

## Setup

Requirements:

- [Torch][10]
- [loadcaffe][9]

Download the MSCOCO train+val images and QA data using `sh data/download_data.sh`.
If you have them downloaded, copy over the `train2014` and `val2014` image folders
and VQA JSON files to the `data` folder.

Download the [VGG-19][7] Caffe model and prototxt using `sh models/download_models.sh`.

### Known issues

- To avoid memory issues with LuaJIT, install Torch with vanilla Lua.
More instructions [here][4].
- If working with plain Lua, [luaffifb][8] may be needed for [loadcaffe][9],
unless using pre-extracted fc7 features.

## Usage

-- Work in progress --

## Implementation Details

- Last hidden layer image features from [VGG19][6]
- [GloVe][5] 200d word embeddings as question features
- For batched implementation of Ren's model, questions of equal word length are grouped
together. So if batch size is 50, and there aren't at least 50 questions of length `n`,
those questions aren't used for training. Yeah, I know..
- Training questions are further filtered for the top-n answers. `n = 1000` by default,
but can be set in `utils/DataLoader.lua`.

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
