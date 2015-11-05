
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
cmd:option('-data_dir', 'data', 'data directory.')

-- model params
cmd:option('-rnn_size', 256, 'size of LSTM internal state')
cmd:option('-embedding_size', 200, 'size of word embeddings')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-batch_size', 64, 'batch size')
-- bookkeeping
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-save_every', 50)
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-feat_layer', 'fc7', 'Layer to extract features from, for input to LSTM')
cmd:option('-input_image_dir', '/srv/share/data/mscoco/images')
-- gpu/cpu
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')

opt = cmd:parse(arg or {})
torch.manualSeed(opt.seed)

loader = DataLoader.create(opt.data_dir, opt.batch_size)

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

vgg = loadcaffe.load(opt.proto_file, opt.model_file, 'nn')
if opt.gpuid >= 0 then
    vgg = vgg:cuda()
end

vgg_fc7 = nn.Sequential()

for i = 1, #vgg.modules do
    local layer = vgg:get(i)
    local name = layer.name
    vgg_fc7:add(layer)
    if name == opt.feat_layer then
        break
    end
end

if opt.gpuid >= 0 then
    vgg_fc7 = vgg_fc7:cuda()
end

-- print(vgg_fc7)

ltw = nn.LookupTable(loader.q_vocab_size, opt.embedding_size)
ltw.weight = loader.q_embeddings:clone()

lti = nn.Linear(4096, opt.embedding_size)

lstm = LSTM.create(opt.embedding_size, opt.rnn_size)

sm = nn.Sequential()
sm:add(nn.Linear(opt.rnn_size, loader.a_vocab_size))
sm:add(nn.LogSoftMax())

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

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

init_state = {}

local h_init = torch.zeros(opt.batch_size, opt.rnn_size)

if opt.gpuid >= 0 then
    h_init = h_init:cuda()
end

table.insert(init_state, h_init:clone())
table.insert(init_state, h_init:clone())

local init_state_global = utils.clone_list(init_state)

-- for i = 1, 2 do
feval = function(x)

    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    data = loader:next_batch()

    clones = {}
    clones = utils.clone_many_times(lstm, loader.counts[loader.count_idx] + 1)

    q_batch = torch.ShortTensor(opt.batch_size, loader.counts[loader.count_idx])
    img_batch = torch.DoubleTensor(opt.batch_size, 3, 224, 224)
    a_batch = torch.ShortTensor(opt.batch_size)

    for j = 1, opt.batch_size do
        local fp = path.join(opt.input_image_dir, string.format('train2014/COCO_train2014_%.12d.jpg', data.image[j]))
        img_batch[j] = utils.preprocess(image.scale(image.load(fp, 3), 224, 224))
        -- print(fp)
        q_batch[j] = data.question[j]
        a_batch[j] = data.answer[j]
    end

    if opt.gpuid >= 0 then
        img_batch = img_batch:cuda()
        q_batch = q_batch:cuda()
        a_batch = a_batch:cuda()
    end

    qf = ltw:forward(q_batch)

    fc7 = vgg_fc7:forward(img_batch)
    imf = lti:forward(fc7)

    ------------ forward pass ------------

    rnn_state = {[0] = init_state_global}
    loss = 0
    lst = clones[1]:forward{imf, unpack(rnn_state[0])}
    rnn_state[1] = {}
    for i = 1, #init_state do table.insert(rnn_state[1], lst[i]) end
    for t = 2, loader.counts[loader.count_idx] + 1 do
        lst = clones[t]:forward{qf:select(2,t-1), unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
    end

    prediction = sm:forward(lst[#lst])
    loss = criterion:forward(prediction, a_batch)

    print('loss: ' .. loss)

    ------------ backward pass ------------

    dloss = criterion:backward(prediction, a_batch)
    doutput_t = sm:backward(lst[#lst], dloss)

    drnn_state = {[loader.counts[loader.count_idx] + 1] = utils.clone_list(init_state, true)}
    drnn_state[loader.counts[loader.count_idx] + 1][2] = doutput_t

    for t = loader.counts[loader.count_idx] + 1, 2, -1 do
        dlst = clones[t]:backward({qf:select(2, t-1), unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        table.insert(drnn_state[t-1], dlst[2])
        table.insert(drnn_state[t-1], dlst[3])
    end

    _, lstm_dparams = lstm:getParameters()
    lstm_dparams:clamp(-5, 5)

    return loss, params

end

local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}

epochs = 1e2
losses = {}
for i = 1, epochs do
    _,local_loss = optim.rmsprop(feval, params, optim_state)
    losses[#losses + 1] = local_loss[1]
end