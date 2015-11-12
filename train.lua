--[[
    Torch implementation of the VIS + LSTM model from the paper
    'Exploring Models and Data for Image Question Answering'
    by Mengye Ren, Ryan Kiros & Richard Zemel
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'image'

local utils = require 'utils.misc'
local DataLoader = require 'utils.DataLoader'

require 'loadcaffe_wrapper'
local LSTM = require 'lstm'

cmd = torch.CmdLine()
cmd:text('Options')

-- model params
cmd:option('-rnn_size', 1024, 'Size of LSTM internal state')
cmd:option('-embedding_size', 200, 'Size of word embeddings')
-- optimization
cmd:option('-learning_rate', 5e-4, 'Learning rate')
cmd:option('-learning_rate_decay', 0.95, 'Learning rate decay')
cmd:option('-learning_rate_decay_after', 10, 'In number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.95, 'Decay rate for rmsprop')
cmd:option('-batch_size', 64, 'Batch size')
cmd:option('-max_epochs', 50, 'Number of full passes through the training data')
cmd:option('-dropout', 0.5, 'Dropout')
cmd:option('-init_from', '', 'Initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-save_every', 1000, 'No. of iterations after which to checkpoint')
cmd:option('-train_fc7_file', 'data/train_fc7.t7', 'Path to fc7 features of training set')
cmd:option('-train_fc7_image_id_file', 'data/train_fc7_image_id.t7', 'Path to fc7 image ids of training set')
cmd:option('-val_fc7_file', 'data/val_fc7.t7', 'Path to fc7 features of validation set')
cmd:option('-val_fc7_image_id_file', 'data/val_fc7_image_id.t7', 'Path to fc7 image ids of validation set')
cmd:option('-data_dir', 'data', 'Data directory')
cmd:option('-checkpoint_dir', 'checkpoints', 'Checkpoint directory')
cmd:option('-savefile', 'vqa', 'Filename to save checkpoint to')
-- gpu/cpu
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')

opt = cmd:parse(arg or {})
torch.manualSeed(opt.seed)

require 'vtutils'
opt.gpuid = obtain_gpu_lock_id.get_id()

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

local loader = DataLoader.create(opt.data_dir, opt.batch_size, opt)

local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('Loading model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    local protos = checkpoint.protos
    ltw = protos.ltw
    lti = protos.lti
    lstm = protos.lstm
    sm = protos.sm
    do_random_init = false
else
    ltw = nn.Sequential()
    ltw:add(nn.LookupTable(loader.q_vocab_size, opt.embedding_size))
    ltw:get(1).weight = loader.q_embeddings:clone()
    ltw:add(nn.Dropout(opt.dropout))

    lti = nn.Sequential()
    lti:add(nn.Linear(4096, opt.embedding_size))
    lti:add(nn.Tanh())
    lti:add(nn.Dropout(opt.dropout))

    lstm = LSTM.create(opt.embedding_size, opt.rnn_size)

    sm = nn.Sequential()
    sm:add(nn.Linear(opt.rnn_size, loader.a_vocab_size))
    sm:add(nn.LogSoftMax())
end

-- model combines all 'trainable' parameters
model = nn.Sequential()
model:add(lti)
model:add(lstm)
model:add(sm)

criterion = nn.ClassNLLCriterion()

if opt.gpuid >= 0 then
    ltw = ltw:cuda()
    lti = lti:cuda()
    lstm = lstm:cuda()
    sm = sm:cuda()
    model = model:cuda()
    criterion = criterion:cuda()
end

-- put the above things into one flattened parameters tensor
params, grad_params = utils.combine_all_parameters(model)

print('Parameters: ' .. params:size(1))
print('Batches: ' .. loader.batch_data.train.nbatches)

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08)
end

init_state = {}

local h_init = torch.zeros(opt.batch_size, opt.rnn_size)

if opt.gpuid >= 0 then
    h_init = h_init:cuda()
end

table.insert(init_state, h_init:clone())
table.insert(init_state, h_init:clone())

clones = {}
clones = utils.clone_many_times(lstm, loader.q_max_length + 1)

local init_state_global = utils.clone_list(init_state)

feval_val = function(max_batches)

    count = 0
    n = loader.batch_data.val.nbatches
    if max_batches ~= nil then n = math.min(n, max_batches) end

    for i = 1, n do

        q_batch, a_batch, i_batch = loader:next_batch('val')

        qf = ltw:forward(q_batch)

        imf = lti:forward(i_batch)

        if opt.gpuid >= 0 then
            imf = imf:cuda()
        end

        loss = 0
        rnn_state = {[0] = init_state_global}

        for t = 1, loader.q_max_length do
            lst = clones[t]:forward{qf:select(2,t), unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
        end

        lst = clones[loader.q_max_length + 1]:forward{imf, unpack(rnn_state[loader.q_max_length])}

        prediction = sm:forward(lst[#lst])

        _, idx  = prediction:max(2)
        for j = 1, opt.batch_size do
            if idx[j][1] == a_batch[j] then
                count = count + 1
            end
        end

    end

    return count / (n * opt.batch_size)

end

feval = function(x)

    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    q_batch, a_batch, i_batch = loader:next_batch()

    qf = ltw:forward(q_batch)

    imf = lti:forward(i_batch)

    if opt.gpuid >= 0 then
        imf = imf:cuda()
    end

    ------------ forward pass ------------

    loss = 0
    rnn_state = {[0] = init_state_global}

    for t = 1, loader.q_max_length do
        lst = clones[t]:forward{qf:select(2,t), unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
    end

    lst = clones[loader.q_max_length + 1]:forward{imf, unpack(rnn_state[loader.q_max_length])}

    prediction = sm:forward(lst[#lst])

    loss = criterion:forward(prediction, a_batch)

    ------------ backward pass ------------

    dloss = criterion:backward(prediction, a_batch)
    doutput_t = sm:backward(lst[#lst], dloss)

    drnn_state = {[loader.q_max_length + 1] = utils.clone_list(init_state, true)}
    drnn_state[loader.q_max_length + 1][2] = doutput_t

    dlst = clones[loader.q_max_length + 1]:backward({imf, unpack(rnn_state[loader.q_max_length])}, drnn_state[loader.q_max_length + 1])

    drnn_state[loader.q_max_length] = {}
    table.insert(drnn_state[loader.q_max_length], dlst[2])
    table.insert(drnn_state[loader.q_max_length], dlst[3])

    lti:backward(i_batch, dlst[1])

    for t = loader.q_max_length, 1, -1 do
        dlst = clones[t]:backward({qf:select(2, t), unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        table.insert(drnn_state[t-1], dlst[2])
        table.insert(drnn_state[t-1], dlst[3])
    end

    _, lstm_dparams = lstm:getParameters()
    lstm_dparams:clamp(-5, 5)

    return loss, grad_params

end

local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}

losses = {}
iterations = opt.max_epochs * loader.batch_data.train.nbatches
lloss = 0
for i = 1, iterations do
    _,local_loss = optim.rmsprop(feval, params, optim_state)
    losses[#losses + 1] = local_loss[1]

    lloss = lloss + local_loss[1]
    local epoch = i / loader.batch_data.train.nbatches

    if i%10 == 0 then
        print('epoch ' .. epoch .. ' loss ' .. lloss / 10)
        lloss = 0
    end

    if i % loader.batch_data.train.nbatches == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    if i % opt.save_every == 0 or i == iterations then
        print('Checkpointing. Calculating validation accuracy..')
        local val_acc = feval_val()
        local savefile = string.format('%s/%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_acc)
        print('Saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.opt = opt
        checkpoint.protos = {}
        checkpoint.protos.ltw = ltw
        checkpoint.protos.lti = lti
        checkpoint.protos.lstm = lstm
        checkpoint.protos.sm = sm
        checkpoint.vocab_size = loader.q_vocab_size
        torch.save(savefile, checkpoint)
    end

    if i%10 == 0 then
        collectgarbage()
    end
end
