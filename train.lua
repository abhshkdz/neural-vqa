--[[
    Torch implementation of the VIS + LSTM model from the paper
    'Exploring Models and Data for Image Question Answering'
    by Mengye Ren, Ryan Kiros & Richard Zemel.

    This implementation passes the question embeddings
    first and then the image embeddings into the LSTM,
    and does a softmax over the answer vocabulary.
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

local utils = require 'utils.misc'
local DataLoader = require 'utils.DataLoader'

local LSTM = require 'lstm'

cmd = torch.CmdLine()
cmd:text('Options')

-- model params
cmd:option('-rnn_size', 512, 'Size of LSTM internal state')
cmd:option('-num_layers', 2, 'Number of layers in LSTM')
cmd:option('-embedding_size', 512, 'Size of word embeddings')
-- optimization
cmd:option('-learning_rate', 4e-4, 'Learning rate')
cmd:option('-learning_rate_decay', 0.95, 'Learning rate decay')
cmd:option('-learning_rate_decay_after', 15, 'In number of epochs, when to start decaying the learning rate')
cmd:option('-alpha', 0.8, 'alpha for adam')
cmd:option('-beta', 0.999, 'beta used for adam')
cmd:option('-epsilon', 1e-8, 'epsilon that goes into denominator for smoothing')
cmd:option('-batch_size', 200, 'Batch size')
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

-- parse command-line parameters
opt = cmd:parse(arg or {})
print(opt)
torch.manualSeed(opt.seed)

-- gpu stuff
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

-- initialize the data loader
-- checks if .t7 data files exist
-- if they don't or if they're old,
-- they're created from scratch and loaded
local loader = DataLoader.create(opt.data_dir, opt.batch_size, opt)

-- create the directory for saving snapshots of model at different times during training
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

local do_random_init = true
if string.len(opt.init_from) > 0 then
    -- initializing model from checkpoint
    print('Loading model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    do_random_init = false
else
    -- model definition
    -- components: ltw, lti, lstm and sm
    protos = {}

    -- ltw: lookup table + dropout for question words
    -- each word of the question gets mapped to its index in vocabulary
    -- and then is passed through ltw to get a vector of size `embedding_size`
    -- lookup table dimensions are `vocab_size` x `embedding_size`
    protos.ltw = nn.Sequential()
    protos.ltw:add(nn.LookupTable(loader.q_vocab_size+1, opt.embedding_size))
    protos.ltw:add(nn.Dropout(opt.dropout))

    -- lti: fully connected layer + dropout for image features
    -- activations from the last fully connected layer of the deep convnet (VGG in this case)
    -- are passed through lti to get a vector of `embedding_size`
    -- linear layer dimensions are 4096 (size of fc7 layer) x `embedding_size`
    protos.lti = nn.Sequential()
    protos.lti:add(nn.Linear(4096, opt.embedding_size))
    protos.lti:add(nn.Tanh())
    protos.lti:add(nn.Dropout(opt.dropout))

    -- lstm: long short-term memory cell which takes a vector of size `embedding_size` at every time step
    -- hidden state h_t of LSTM cell in first layer is passed as input x_t of cell in second layer and so on.
    protos.lstm = LSTM.create(opt.embedding_size, opt.rnn_size, opt.num_layers)

    -- sm: linear layer + softmax over the answer vocabulary
    -- linear layer dimensions are `rnn_size` x `answer_vocab_size`
    protos.sm = nn.Sequential()
    protos.sm:add(nn.Linear(opt.rnn_size, loader.a_vocab_size))
    protos.sm:add(nn.LogSoftMax())

    -- negative log-likelihood loss
    protos.criterion = nn.ClassNLLCriterion()

    -- pass over the model to gpu
    if opt.gpuid >= 0 then
        protos.ltw = protos.ltw:cuda()
        protos.lti = protos.lti:cuda()
        protos.lstm = protos.lstm:cuda()
        protos.sm = protos.sm:cuda()
        protos.criterion = protos.criterion:cuda()
    end
end

-- put all trainable model parameters into one flattened parameters tensor
params, grad_params = utils.combine_all_parameters(protos.lti, protos.lstm, protos.sm)

print('Parameters: ' .. params:size(1))
print('Batches: ' .. loader.batch_data.train.nbatches)

-- initialize model parameters
if do_random_init then
    params:uniform(-0.08, 0.08)
end

-- make clones of the LSTM model that shared parameters for subsequent timesteps (unrolling)
lstm_clones = {}
lstm_clones = utils.clone_many_times(protos.lstm, loader.q_max_length + 1)

-- initialize h_0 and c_0 of LSTM to zero tensors and store in `init_state`
init_state = {}
for L = 1, opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- make a clone of `init_state` as it's going to be modified later
local init_state_global = utils.clone_list(init_state)

-- closure to calculate accuracy over validation set
feval_val = function(max_batches)

    count = 0
    n = loader.batch_data.val.nbatches

    -- set `n` to `max_batches` if provided
    if max_batches ~= nil then n = math.min(n, max_batches) end

    -- set to evaluation mode for dropout to work properly
    protos.ltw:evaluate()
    protos.lti:evaluate()

    for i = 1, n do

        -- load question batch, answer batch and image features batch
        q_batch, a_batch, i_batch = loader:next_batch('val')

        -- 1st index of `nn.LookupTable` is reserved for zeros
        q_batch = q_batch + 1

        -- forward the question features through ltw
        qf = protos.ltw:forward(q_batch)

        -- forward the image features through lti
        imf = protos.lti:forward(i_batch)

        -- convert to CudaTensor if using gpu
        if opt.gpuid >= 0 then
            imf = imf:cuda()
        end

        -- set the state at 0th time step of LSTM
        rnn_state = {[0] = init_state_global}

        -- LSTM forward pass for question features
        for t = 1, loader.q_max_length do
            lst = lstm_clones[t]:forward{qf:select(2,t), unpack(rnn_state[t-1])}
            -- at every time step, set the rnn state (h_t, c_t) to be passed as input in next time step
            rnn_state[t] = {}
            for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
        end

        -- after completing the unrolled LSTM forward pass with question features, forward the image features
        lst = lstm_clones[loader.q_max_length + 1]:forward{imf, unpack(rnn_state[loader.q_max_length])}

        -- forward the hidden state at the last time step to get softmax over answers
        prediction = protos.sm:forward(lst[#lst])

        -- count number of correct answers
        _, idx  = prediction:max(2)
        for j = 1, opt.batch_size do
            if idx[j][1] == a_batch[j] then
                count = count + 1
            end
        end

    end

    -- set to training mode once done with validation
    protos.ltw:training()
    protos.lti:training()

    -- return accuracy
    return count / (n * opt.batch_size)

end

-- closure to run a forward and backward pass and return loss and gradient parameters
feval = function(x)

    -- get latest parameters
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    -- load question batch, answer batch and image features batch
    q_batch, a_batch, i_batch = loader:next_batch()

    -- slightly hackish; 1st index of `nn.LookupTable` is reserved for zeros
    q_batch = q_batch + 1

    -- forward the question features through ltw
    qf = protos.ltw:forward(q_batch)

    -- forward the image features through lti
    imf = protos.lti:forward(i_batch)

    -- convert to CudaTensor if using gpu
    if opt.gpuid >= 0 then
        imf = imf:cuda()
    end

    ------------ forward pass ------------

    -- set initial loss
    loss = 0

    -- set the state at 0th time step of LSTM
    rnn_state = {[0] = init_state_global}

    -- LSTM forward pass for question features
    for t = 1, loader.q_max_length do
        lst = lstm_clones[t]:forward{qf:select(2,t), unpack(rnn_state[t-1])}
        -- at every time step, set the rnn state (h_t, c_t) to be passed as input in next time step
        rnn_state[t] = {}
        for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
    end

    -- after completing the unrolled LSTM forward pass with question features, forward the image features
    lst = lstm_clones[loader.q_max_length + 1]:forward{imf, unpack(rnn_state[loader.q_max_length])}

    -- forward the hidden state at the last time step to get softmax over answers
    prediction = protos.sm:forward(lst[#lst])

    -- calculate loss
    loss = protos.criterion:forward(prediction, a_batch)

    ------------ backward pass ------------

    -- backprop through loss and softmax
    dloss = protos.criterion:backward(prediction, a_batch)
    doutput_t = protos.sm:backward(lst[#lst], dloss)

    -- set internal state of LSTM (starting from last time step)
    drnn_state = {[loader.q_max_length + 1] = utils.clone_list(init_state, true)}
    drnn_state[loader.q_max_length + 1][opt.num_layers * 2] = doutput_t

    -- backprop for last time step (image features)
    dlst = lstm_clones[loader.q_max_length + 1]:backward({imf, unpack(rnn_state[loader.q_max_length])}, drnn_state[loader.q_max_length + 1])

    -- backprop into image linear layer
    protos.lti:backward(i_batch, dlst[1])

    -- set LSTM state
    drnn_state[loader.q_max_length] = {}
    for i,v in pairs(dlst) do
        if i > 1 then
            drnn_state[loader.q_max_length][i-1] = v
        end
    end

    dqf = torch.Tensor(qf:size()):zero()
    if opt.gpuid >= 0 then
        dqf = dqf:cuda()
    end

    -- backprop into the LSTM for rest of the time steps
    for t = loader.q_max_length, 1, -1 do
        dlst = lstm_clones[t]:backward({qf:select(2, t), unpack(rnn_state[t-1])}, drnn_state[t])
        dqf:select(2, t):copy(dlst[1])
        drnn_state[t-1] = {}
        for i,v in pairs(dlst) do
            if i > 1 then
                drnn_state[t-1][i-1] = v
            end
        end
    end

    -- zero gradient buffers of lookup table, backprop into it and update parameters
    protos.ltw:zeroGradParameters()
    protos.ltw:backward(q_batch, dqf)
    protos.ltw:updateParameters(opt.learning_rate)

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params

end

-- optim state with ADAM parameters
local optim_state = {learningRate = opt.learning_rate, alpha = opt.alpha, beta = opt.beta, epsilon = opt.epsilon}

-- train / val loop!
losses = {}
iterations = opt.max_epochs * loader.batch_data.train.nbatches
print('Max iterations: ' .. iterations)
lloss = 0
for i = 1, iterations do
    _, local_loss = optim.adam(feval, params, optim_state)

    losses[#losses + 1] = local_loss[1]

    lloss = lloss + local_loss[1]
    local epoch = i / loader.batch_data.train.nbatches

    if i%10 == 0 then
        print('epoch ' .. epoch .. ' loss ' .. lloss / 10)
        lloss = 0
    end

    -- Decay learning rate occasionally
    if i % loader.batch_data.train.nbatches == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- Calculate validation accuracy and save model snapshot
    if i % opt.save_every == 0 or i == iterations then
        print('Checkpointing. Calculating validation accuracy..')
        local val_acc = feval_val()
        local savefile = string.format('%s/%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_acc)
        print('Saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.opt = opt
        checkpoint.protos = protos
        checkpoint.vocab_size = loader.q_vocab_size
        torch.save(savefile, checkpoint)
    end

    if i%10 == 0 then
        collectgarbage()
    end
end
