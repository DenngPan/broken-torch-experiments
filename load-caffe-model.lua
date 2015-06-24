require 'dp'
require 'optim'
require 'cunn'
require 'TrainHelpers'
require 'cutorch'
require 'loadcaffe'
require 'cudnn'

dataPath = paths.concat(dp.DATA_DIR, 'ImageNet')
trainPath = paths.concat(dataPath, 'ILSVRC2012_img_train')
validPath = paths.concat(dataPath, 'ILSVRC2012_img_val')
metaPath = paths.concat(dataPath, 'metadata')

--[[Model]]--
model = loadcaffe.load('/mnt/caffe/models/bvlc_alexnet/deploy.prototxt',
    '/mnt/caffe/models/bvlc_alexnet/caffe_alexnet_train_iter_1.caffemodel',
    'cudnn')
-- Replace SoftMax with LogSoftMax
model.modules[#model.modules] = nil
model:add(nn.LogSoftMax():cuda())

cutorch.setDevice(1)
--model:cuda()

loss = nn.ClassNLLCriterion()
loss:cuda()

-- -- Fill weights, learning rates, and biases
-- for i,layer in ipairs(model.modules) do
--    if layer.bias then
--       layer.bias:fill(0)
--       layer.weight:normal(0, 0.05)
--    end
-- end
-- ---- set individual learning rates and weight decays
local wds = 1e-4

weights, gradients = model:getParameters()
weight_size = weights:size(1)
learningRates = torch.Tensor(weight_size):fill(0)
weightDecays = torch.Tensor(weight_size):fill(0)
counter = 0
for i, layer in ipairs(model.modules) do
   if layer.__typename == 'cudnn.SpatialConvolution' then
      local base_lr = 1.0
      if layer.tag == 'lastlayer' then
          base_lr = 0.1 -- slower learning for the last layer please
          print("Picking a lower learning rate for the last layers")
      end
      local weight_size = layer.weight:numel()  -- :size(1)*layer.weight:size(2)
      learningRates[{{counter+1, counter+weight_size}}]:fill(1 * base_lr)
      weightDecays[{{counter+1, counter+weight_size}}]:fill(wds)
      counter = counter+weight_size
      local bias_size = layer.bias:numel() -- :size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(2 * base_lr)
      weightDecays[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
  end
end
print("Filling at counter", counter,"but have", weight_size,"total")
for i, layer in ipairs(model.modules) do
   if layer.__typename == 'nn.Linear' then
      local weight_size = layer.weight:numel()  -- :size(1)*layer.weight:size(2)
      local bias_size = layer.bias:numel() -- :size(1)
      counter = counter + weight_size + bias_size
   end
end
print("Now accounted for counter", counter,"but have", weight_size,"total")

--[[data]]--
ds_all = dp.ImageNet{
   train_path=trainPath, valid_path=validPath, meta_path=metaPath,
   verbose=true,
}

function eval(inputs, targets)
   model:training()
   inputs = inputs:cuda()
   targets = targets:cuda()
   gradients:zero() -- should i do this instead...?
   local y = model:forward(inputs)
   local loss_val = loss:forward(y, targets)
   local df_dw = loss:backward(y, targets)
   model:backward(inputs, df_dw)
   return loss_val, gradients
end


-- Set up dataset
ds_train = ds_all:loadTrain()
preprocess = TrainHelpers.normalizePreprocessDataset(ds_train)
ds_val = ds_all:loadValid()
sgdState = {
    learningRate = 0.01,
    momentum = 0.9,
    weightDecay = 0.0005,
    learningRateDecay = 0,
    learningRates = learningRates,
    weightDecays = weightDecays
}

-- Sample one epoch!
exp = TrainHelpers.ExperimentHelper{
   model = model,
   trainDataset = ds_train,
   batchSize = 192,
   preprocessFunc = preprocess,
   datasetMultithreadLoading = 4,
   sgdState = sgdState,
   eval = eval,
}
exp:printEpochProgress{everyNBatches = 1}
exp:printAverageTrainLoss{everyNBatches = 20}
exp:snapshotModel{everyNBatches = 1000,
   filename="alexnet-%s.t7"
}
exp:trainForever(eval, weights)
