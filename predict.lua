
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'image'

local utils = require 'utils.misc'
local DataLoader = require 'utils.DataLoader'

require 'loadcaffe'
local LSTM = require 'lstm'

cmd = torch.CmdLine()
cmd:text('Options')

-- model params
cmd:option('-rnn_size', 512, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'Number of layers in LSTM')
cmd:option('-embedding_size', 512, 'size of word embeddings')
-- optimization
cmd:option('-batch_size', 64, 'batch size')
-- bookkeeping
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-checkpoint_file', 'checkpoints/vqa_epoch23.26_0.4610.t7', 'Checkpoint file to use for predictions')
cmd:option('-data_dir', 'data', 'data directory')
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-feat_layer', 'fc7', 'Layer to extract features from')
cmd:option('-train_fc7_file', 'data/train_fc7.t7', 'Path to fc7 features of training set')
cmd:option('-train_fc7_image_id_file', 'data/train_fc7_image_id.t7', 'Path to fc7 image ids of training set')
cmd:option('-val_fc7_file', 'data/val_fc7.t7', 'Path to fc7 features of validation set')
cmd:option('-val_fc7_image_id_file', 'data/val_fc7_image_id.t7', 'Path to fc7 image ids of validation set')
cmd:option('-input_image_path', 'data/train2014/COCO_train2014_000000405541.jpg', 'Image path')
cmd:option('-question', 'What is the cat on?', 'Question string')
-- gpu/cpu
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')

opt = cmd:parse(arg or {})
torch.manualSeed(opt.seed)

if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- torch is 1-indexed
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back to CPU mode')
        opt.gpuid = -1
    end
end

loader = DataLoader.create(opt.data_dir, opt.batch_size, opt, 'predict')

-- load model checkpoint

print('loading checkpoint from ' .. opt.checkpoint_file)
checkpoint = torch.load(opt.checkpoint_file)

lstm_clones = {}
lstm_clones = utils.clone_many_times(checkpoint.protos.lstm, loader.q_max_length + 1)

checkpoint.protos.ltw:evaluate()
checkpoint.protos.lti:evaluate()

q_vocab_size = checkpoint.vocab_size

a_iv = {}
for i,v in pairs(loader.a_vocab_mapping) do
    a_iv[v] =  i
end

q_iv = {}
for i,v in pairs(loader.q_vocab_mapping) do
    q_iv[v] =  i
end

if q_vocab_size ~= loader.q_vocab_size then
    print('Vocab size of checkpoint and data are different.')
end

cnn = loadcaffe.load(opt.proto_file, opt.model_file)

function predict(input_image_path, question_string)

    -- extract image features

    if opt.gpuid >= 0 then
        cnn = cnn:cuda()
    end

    local cnn_fc7 = nn.Sequential()

    for i = 1, #cnn.modules do
        local layer = cnn:get(i)
        local name = layer.name
        cnn_fc7:add(layer)
        if name == opt.feat_layer then
            break
        end
    end

    if opt.gpuid >= 0 then
        cnn_fc7 = cnn_fc7:cuda()
    end

    local img = utils.preprocess(image.scale(image.load(input_image_path, 3), 224, 224))

    if opt.gpuid >= 0 then
        img = img:cuda()
    end

    local fc7 = cnn_fc7:forward(img)
    local imf = checkpoint.protos.lti:forward(fc7)

    -- extract question features

    local question = torch.ShortTensor(loader.q_max_length):zero()

    local idx = 1
    local words = {}
    for token in string.gmatch(question_string, "%a+") do
        words[idx] = token
        idx = idx + 1
    end

    for i = 1, #words do
        question[loader.q_max_length - #words + i] = loader.q_vocab_mapping[words[i]] or loader.q_vocab_mapping['UNK']
    end

    if opt.gpuid >= 0 then
        question = question:cuda()
    end

    -- 1st index of `nn.LookupTable` is for zeros
    question = question + 1

    local qf = checkpoint.protos.ltw:forward(question)

    -- lstm + softmax

    local init_state = {}
    for L = 1, opt.num_layers do
        local h_init = torch.zeros(1, opt.rnn_size)
        if opt.gpuid >=0 then h_init = h_init:cuda() end
        table.insert(init_state, h_init:clone())
        table.insert(init_state, h_init:clone())
    end

    local rnn_state = {[0] = init_state}

    for t = 1, loader.q_max_length do
        local lst = lstm_clones[t]:forward{qf:select(1,t):view(1,-1), unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
    end

    local lst = lstm_clones[loader.q_max_length + 1]:forward{imf:view(1,-1), unpack(rnn_state[loader.q_max_length])}

    local prediction = checkpoint.protos.sm:forward(lst[#lst])

    local _, idx  = prediction:max(2)

    print(a_iv[idx[1][1]])
end

predict(opt.input_image_path, opt.question)
