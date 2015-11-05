
require 'torch'
require 'nn'
require 'image'

local utils = require 'utils.misc'
local DataLoader = require 'utils.DataLoader'

require 'loadcaffe_wrapper'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-batch_size', 64, 'batch size')
-- gpu/cpu
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')
-- bookkeeping
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-data_dir', 'data', 'data directory.')
cmd:option('-feat_layer', 'fc7', 'Layer to extract features from, for input to LSTM')
cmd:option('-input_image_dir', '/srv/share/data/mscoco/images')

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

vgg = loadcaffe.load(opt.proto_file, opt.model_file)
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

tmp_image_id = {}
for i = 1, #loader.data.train do
    tmp_image_id[loader.data.train[i].image_id] = 1
end

image_id = {}
idx = 1
for i, v in pairs(tmp_image_id) do
    image_id[idx] = i
    idx = idx + 1
end

fc7 = {}
idx = 1

repeat
    -- local timer = torch.Timer()
    img_batch = torch.DoubleTensor(opt.batch_size, 3, 224, 224)
    img_id_batch = {}
    for i = 1, opt.batch_size do
        if not loader.data.train[idx] then
            break
        end
        local fp = path.join(opt.input_image_dir, string.format('train2014/COCO_train2014_%.12d.jpg', image_id[idx]))
        img_batch[i] = utils.preprocess(image.scale(image.load(fp, 3), 224, 224))
        img_id_batch[i] = image_id[idx]
        idx = idx + 1
    end

    if opt.gpuid >= 0 then
        img_batch = img_batch:cuda()
    end

    fc7_batch = vgg_fc7:forward(img_batch)
    for i = 1, #img_id_batch do
        fc7[img_id_batch[i]] = fc7_batch[i]:float()
    end

    -- local time = timer:time().real
    print(idx-1 .. '/' .. #image_id)
until idx >= #image_id

torch.save(path.join(opt.data_dir, 'fc7.t7'), fc7)