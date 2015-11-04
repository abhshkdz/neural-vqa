local utils = {}

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function utils.preprocess(img)
    local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):mul(256.0)
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img:add(-1, mean_pixel)
    return img
end


-- Undo the above preprocessing.
function utils.deprocess(img)
    local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img = img + mean_pixel
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):div(256.0)
    return img
end

return utils