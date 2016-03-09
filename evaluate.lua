require 'torch'
require 'nn'
require 'nngraph'

local utils = require 'utils.misc'
local DataLoader = require 'utils.DataLoader'

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
cmd:option('-checkpoint_file', 'checkpoints/vqa_epoch23.26_0.4610.t7', 'Checkpoint file to use for predictions')
cmd:option('-data_dir', 'data', 'data directory')
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-train_fc7_file', 'data/train_fc7.t7', 'Path to fc7 features of training set')
cmd:option('-train_fc7_image_id_file', 'data/train_fc7_image_id.t7', 'Path to fc7 image ids of training set')
cmd:option('-val_fc7_file', 'data/val_fc7.t7', 'Path to fc7 features of validation set')
cmd:option('-val_fc7_image_id_file', 'data/val_fc7_image_id.t7', 'Path to fc7 image ids of validation set')
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

loader = DataLoader.create(opt.data_dir, opt.batch_size, opt)

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

init_state = {}
for L = 1, opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

local init_state_global = utils.clone_list(init_state)

count = 0
for i = 1, loader.batch_data.val.nbatches do
    q_batch, a_batch, i_batch = loader:next_batch('val')

    -- 1st index of `nn.LookupTable` is for zeros
    q_batch = q_batch + 1

    qf = checkpoint.protos.ltw:forward(q_batch)

    imf = checkpoint.protos.lti:forward(i_batch)

    if opt.gpuid >= 0 then
        imf = imf:cuda()
    end

    rnn_state = {[0] = init_state_global}

    for t = 1, loader.q_max_length do
        lst = lstm_clones[t]:forward{qf:select(2,t), unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
    end

    lst = lstm_clones[loader.q_max_length + 1]:forward{imf, unpack(rnn_state[loader.q_max_length])}

    prediction = checkpoint.protos.sm:forward(lst[#lst])

    _, idx  = prediction:max(2)
    for j = 1, opt.batch_size do
        if idx[j][1] == a_batch[j] then
            count = count + 1
        end
    end

    print(count .. '/' .. i * opt.batch_size)
end

print(count / (loader.batch_data.val.nbatches * opt.batch_size))
