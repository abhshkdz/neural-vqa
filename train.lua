
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
cmd:option('-batch_size', 64, '')

cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')

cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')

-- cmd:option('-damp_layer', 'conv5_4', 'Layer to extract activations from, for damping using attention maps')
cmd:option('-feat_layer', 'fc7', 'Layer to extract features from, for input to LSTM')

cmd:option('-input_image_dir', '/srv/share/data/mscoco/images')
cmd:option('-blur_image_dir', '/srv/share/visualAttention/data/analysis/blur_images_png')
cmd:option('-map_image_dir', '/srv/share/visualAttention/data/analysis/map_images_png')

cmd:option('-save_every', 50)

opt = cmd:parse(arg or {})

loader = DataLoader.create(opt.data_dir, opt.batch_size)

-- gpu stuff
-- require 'vtutils'
-- opt.gpuid = obtain_gpu_lock_id.get_id()

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
    vgg:cuda()
end

-- data_dir = "/srv/share/data/mscoco/images/val2014" for i = 1, #data.val do fp = path.join(data_dir, string.format("COCO_val2014_%.12d.jpg", data.val[i].image_id)) if not path.exists(fp) then print(fp) end end
-- data_dir = "/srv/share/data/mscoco/images/train2014" for i = 1, #data.train do fp = path.join(data_dir, string.format("COCO_train2014_%.12d.jpg", data.train[i].image_id)) if not path.exists(fp) then print(fp) end end

local vgg_fc7 = nn.Sequential()

for i = 1, #vgg.modules do
    local layer = vgg:get(i)
    local name = layer.name
    vgg_fc7:add(layer)
    if name == opt.feat_layer then
        break
    end
end

print(vgg_fc7)

local data = loader:next_batch()

for i = 1, 2 do
    local fp = path.join(opt.input_image_dir, string.format('train2014/COCO_train2014_%.12d.jpg', data.image[i]))
    local img = utils.preprocess(image.scale(image.load(fp, 3), 224, 224))
    if opt.gpuid >= 0 then
        img = img:cuda()
    end
    local fc7 = vgg_fc7:forward(img)
end


-- Keep track of which layer to split the two nets at
-- 1st net: input to opt.damp_layer
-- 2nd net: opt.damp_layer + 1 to opt.feat_layer
-- damp_feat_switch = 0

-- damp_net = nn.Sequential()
-- feat_net = nn.Sequential()

-- if opt.gpuid >= 0 then
--     damp_net:cuda()
--     feat_net:cuda()
-- end

-- -- Iterate through the layers and
-- -- split into two nets
-- for i = 1, #vgg do
--     local layer = vgg:get(i)
--     local name = layer.name
--     if damp_feat_switch == 0 then
--         damp_net:add(layer)
--     else
--         feat_net:add(layer)
--     end
--     if name == opt.damp_layer then
--         damp_feat_switch = 1
--     elseif name == opt.feat_layer then
--         break
--     end
-- end

-- local checkpoint = {}

-- for i = 1, #data_loader.data do
--     -- Pass the input image through damp_net
--     local input_image = image.load(data_loader.data[i].input_image, 3)
--     local input_image_scaled = image.scale(input_image, 224, 224)
--     local input_image_caffe = utils.preprocess(input_image_scaled)
--     if opt.gpuid >= 0 then
--         input_image_caffe = input_image_caffe:cuda()
--     end
--     local input_image_activations = damp_net:forward(input_image_caffe)

--     -- Scale map image and damp input image activations
--     local map_image = image.load(data_loader.data[i].map_image, 3)
--     local map_image_scaled = image.scale(map_image, input_image_activations:size()[3], input_image_activations:size()[2])
--     if opt.gpuid >= 0 then
--         map_image_scaled = map_image_scaled:cuda()
--     end
--     for i = 1, input_image_activations:size()[1] do
--         input_image_activations[i]:cmul(map_image_scaled[1])
--     end

--     -- Extract 4096-d fc7 features
--     local image_features = feat_net:forward(input_image_activations)

--     table.insert(checkpoint, {['annotation_id'] = data_loader.data[i].annotation_id, ['vgg_fc7'] = image_features})

--     if i % opt.save_every == 0 then
--         torch.save('savefile', checkpoint)
--     end
-- end
