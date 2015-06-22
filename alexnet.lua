require 'dp'
require 'optim'
require 'cunn'
require 'TrainHelpers'
require 'cutorch'
require 'cudnn'

opt = {
   -- Path to ImageNet
   dataPath = paths.concat(dp.DATA_DIR, 'ImageNet'),
   -- overwrite cache? (SLOW! BE CAREFUL!)
   overwrite = false,
   -- Learning schedule parameters
   learningRate = 0.01,
   --schedule = {[1]=1e-2,[19]=5e-3,[30]=1e-3,[44]=5e-4,[53]=1e-4},
   -- Weight decay
   weightDecay = 0.0, --005,
   momentum = 0.9,
   dampening = 0,
   nesterov = true,
   -- CUDA devices
   cuda = true,
   useDevice = 1,
   -- Batch size
   batchSize = 192
}

opt.trainPath = opt.trainPath or paths.concat(opt.dataPath, 'ILSVRC2012_img_train')
opt.validPath = opt.validPath or paths.concat(opt.dataPath, 'ILSVRC2012_img_val')
opt.metaPath = opt.metaPath or paths.concat(opt.dataPath, 'metadata')

print(opt)


--[[Model]]--

model = nn.Sequential()
model:add(cudnn.SpatialConvolution(3,96,11,11,4,4,2,2))       -- 224 -> 55
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27

model:add(cudnn.SpatialConvolution(96,256,5,5,1,1,2,2))       --  27 -> 27
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13

model:add(cudnn.SpatialConvolution(256,384,3,3,1,1,1,1))      --  13 ->  13
model:add(cudnn.ReLU())

model:add(cudnn.SpatialConvolution(384,384,3,3,1,1,1,1))      --  13 ->  13
model:add(cudnn.ReLU())

model:add(cudnn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
model:add(cudnn.ReLU())

model:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

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

--[[data]]--
ds_all = dp.ImageNet{
   train_path=opt.trainPath, valid_path=opt.validPath, meta_path=opt.metaPath,
   verbose=true,
   cache_mode = opt.overwrite and 'overwrite' or nil
}


-- CUDA-ize
if opt.cuda then
   cutorch.setDevice(opt.useDevice)
   model:cuda()
   loss:cuda()
   print "Now on CUDA"
end

weights,gradients = model:getParameters() -- be sure to do this AFTER CUDA-izing it!
function eval(inputs, targets)
   model:training()
   if opt.cuda then
       inputs = inputs:cuda()
       targets = targets:cuda()
   end
   gradients:zero() -- should i do this instead...?
   local y = model:forward(inputs)
   local loss_val = loss:forward(y, targets)
   local df_dw = loss:backward(y, targets)
   model:backward(inputs, df_dw)
   return loss_val, gradients
end


-- Set up dataset
preprocess = ds_all:normalizePPF()
ds_train = ds_all:loadTrain()
ds_val = ds_all:loadValid()

-- Sample one epoch!
exp = TrainHelpers.ExperimentHelper{
   model = model,
   trainDataset = ds_train,
   batchSize = opt.batchSize,
   preprocessFunc = preprocess,
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   dampening = opt.dampening,
   nesterov = opt.nesterov,
   datasetMultithreadLoading = 4
}
exp:printEpochProgress{everyNBatches = 1}
exp:printAverageTrainLoss{everyNBatches = 10}
exp:snapshotModel{everyNBatches = 3000,
   filename="alexnet-%s.t7"
}
--exp:trainForever()
