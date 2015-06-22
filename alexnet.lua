require 'dp'
require 'optim'
require 'cunn'
require 'TrainHelpers'
require 'cutorch'
--require 'cudnn'

dataPath = paths.concat(dp.DATA_DIR, 'ImageNet')
trainPath = paths.concat(dataPath, 'ILSVRC2012_img_train')
validPath = paths.concat(dataPath, 'ILSVRC2012_img_val')
metaPath = paths.concat(dataPath, 'metadata')

--[[Model]]--

model = nn.Sequential()
model:add(nn.SpatialConvolution(3,96,11,11,4,4,2,2))       -- 224 -> 55
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27

model:add(nn.SpatialConvolution(96,256,5,5,1,1,2,2))       --  27 -> 27
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13

model:add(nn.SpatialConvolution(256,384,3,3,1,1,1,1))      --  13 ->  13
model:add(nn.ReLU())

model:add(nn.SpatialConvolution(384,384,3,3,1,1,1,1))      --  13 ->  13
model:add(nn.ReLU())

model:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
model:add(nn.ReLU())

model:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

model:add(nn.View(256*6*6))
model:add(nn.Linear(256*6*6, 4096))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 4096))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 1000))
model:add(nn.LogSoftMax())

loss = nn.ClassNLLCriterion()

-- -- Fill weights, learning rates, and biases
-- for i,layer in ipairs(model.modules) do
--    if layer.bias then
--       layer.bias:fill(0)
--       layer.weight:normal(0, 0.05)
--    end
-- end
-- ---- set individual learning rates and weight decays
-- local wds = 1e-4
--
-- local dE, param = model:getParameters()
-- local weight_size = dE:size(1)
-- local learningRates = torch.Tensor(weight_size):fill(0)
-- local weightDecays = torch.Tensor(weight_size):fill(0)
-- local counter = 0
-- for i, layer in ipairs(model.modules) do
--    if layer.__typename == 'nn.SpatialConvolution' then
--       local base_lr = 1.0
--       if layer.tag == 'lastlayer' then
--           base_lr = 0.1 -- slower learning for the last layer please
--           print("Picking a lower learning rate for the last layers")
--       end
--       local weight_size = layer.weight:numel()  -- :size(1)*layer.weight:size(2)
--       learningRates[{{counter+1, counter+weight_size}}]:fill(1 * base_lr)
--       weightDecays[{{counter+1, counter+weight_size}}]:fill(wds)
--       counter = counter+weight_size
--       local bias_size = layer.bias:numel() -- :size(1)
--       learningRates[{{counter+1, counter+bias_size}}]:fill(2 * base_lr)
--       weightDecays[{{counter+1, counter+bias_size}}]:fill(0)
--       counter = counter+bias_size
--   end
-- end
-- print("Filling at counter", counter,"but have", weight_size,"total")

--[[data]]--
ds_all = dp.ImageNet{
   train_path=trainPath, valid_path=validPath, meta_path=metaPath,
   verbose=true,
}

cutorch.setDevice(1)
model:cuda()
loss:cuda()

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
    learningRate = 0.001,
    momentum = 0.9,
    weightDecay = 0,
    learningRateDecay = 0,
    learningRates = learningRates,
    weightDecays = weightDecays
}

-- Sample one epoch!
exp = TrainHelpers.ExperimentHelper{
   model = model,
   trainDataset = ds_train,
   batchSize = 128,
   preprocessFunc = preprocess,
   datasetMultithreadLoading = 4,
   sgdState = sgdState
}
exp:printEpochProgress{everyNBatches = 1}
exp:printAverageTrainLoss{everyNBatches = 10}
exp:snapshotModel{everyNBatches = 3000,
   filename="alexnet-%s.t7"
}
exp:trainForever()
