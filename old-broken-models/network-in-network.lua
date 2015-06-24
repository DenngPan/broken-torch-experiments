require 'dp'
require 'optim'
require 'cunn'
require 'TrainHelpers'
require 'cutorch'
require 'cunn'


opt = {
   -- Path to ImageNet
   dataPath = paths.concat(dp.DATA_DIR, 'ImageNet'),
   -- overwrite cache? (SLOW! BE CAREFUL!)
   overwrite = false,
   -- Learning schedule parameters
   -- Weight decay
   -- Batch size
   batchSize = 128
}
cutorch.setDevice(1)

opt.trainPath = opt.trainPath or paths.concat(opt.dataPath, 'ILSVRC2012_img_train')
opt.validPath = opt.validPath or paths.concat(opt.dataPath, 'ILSVRC2012_img_val')
opt.metaPath = opt.metaPath or paths.concat(opt.dataPath, 'metadata')

print(opt)


-- Original:
-- https://github.com/Aysegul/torch-NetworkInNetwork/

local dropout0 = nn.Dropout(0.5)
local dropout1 = nn.Dropout(0.5)

model = nn.Sequential()

--model:add(nn.Transpose({1,4},{1,3},{1,2}))

model:add(nn.SpatialConvolution(3, 96, 11, 11, 4, 4))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(96, 96, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(96, 96, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3, 3, 2, 2))
-- now has size 96x26x26

model:add(nn.SpatialConvolution(96, 256, 5,5,1,1))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(256, 256, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(256, 256, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3,3, 2,2))
-- now has size 256x10x10

model:add(nn.SpatialConvolution(256, 384, 3, 3, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(384,384,1,1))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(384,384,1,1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3,3, 2,2))
model:add(nn.Dropout(0.5))

model:add(nn.SpatialConvolution(384, 1024, 3,3,1,1))
model.modules[#model.modules].tag='lastlayer'
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(1024, 1024, 1,1,1,1))
model.modules[#model.modules].tag='lastlayer'
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(1024, 1000, 1,1,1,1))
model.modules[#model.modules].tag='lastlayer'
model:add(nn.View(1000))
model:add(nn.LogSoftMax())

for i,layer in ipairs(model.modules) do
   if layer.bias then
      layer.bias:fill(0)
      layer.weight:normal(0, 0.05)
   end
end

model:cuda()
loss = nn.ClassNLLCriterion()

----------------------------------------------------------------------
---- set individual learning rates and weight decays
local wds = 1e-4

local dE, param = model:getParameters()
local weight_size = dE:size(1)
local learningRates = torch.Tensor(weight_size):fill(0)
local weightDecays = torch.Tensor(weight_size):fill(0)
local counter = 0
for i, layer in ipairs(model.modules) do
   if layer.__typename == 'nn.SpatialConvolution' then
      local base_lr = 1.0
      if layer.tag == 'lastlayer' then
          base_lr = 0.1 -- slower learning for the last layer please
          print("Picking a lower learning rate for the last layers")
      end
      local weight_size = layer.weight:size(1)*layer.weight:size(2)
      learningRates[{{counter+1, counter+weight_size}}]:fill(1 * base_lr)
      weightDecays[{{counter+1, counter+weight_size}}]:fill(wds)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(2 * base_lr)
      weightDecays[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
  end
end
loss:cuda()


--[[data]]--
ds_all = dp.ImageNet{
   train_path=opt.trainPath, valid_path=opt.validPath, meta_path=opt.metaPath,
   verbose=true,
   cache_mode = opt.overwrite and 'overwrite' or nil
}

weights,gradients = model:getParameters() -- be sure to do this AFTER CUDA-izing it!
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
    learningRate = 2e-3,
    momentum = 0.9,
    dampening = 0,
    weightDecay = 0,
    learningRateDecay = 0,
    learningRates = learningRates,
    weightDecays = weightDecays
}


-- Sample one epoch!
exp = TrainHelpers.ExperimentHelper{
   model = model,
   trainDataset = ds_train,
   batchSize = opt.batchSize,
   preprocessFunc = preprocess,
   datasetMultithreadLoading = 4,
   sgdState = sgdState,
}
exp:printEpochProgress{everyNBatches = 1}
exp:printAverageTrainLoss{everyNBatches = 10}
exp:snapshotModel{everyNBatches = 3000,
   filename="network-in-network-%s.t7"
}
exp:trainForever()
