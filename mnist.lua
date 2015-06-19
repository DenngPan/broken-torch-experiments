require 'dp'
require 'optim'
require 'TrainHelpers'

opt = {
   -- Path to ImageNet
   dataPath = paths.concat(dp.DATA_DIR, 'ImageNet'),
   -- overwrite cache? (SLOW! BE CAREFUL!)
   overwrite = false,
   -- Learning schedule parameters
   learningRate = 0.01,
   --schedule = {[1]=1e-2,[19]=5e-3,[30]=1e-3,[44]=5e-4,[53]=1e-4},
   -- Weight decay
   --weightDecay = 5e-4,
   momentum = 0.9,
   dampening = 0,
   nesterov = true,
   -- CUDA devices
   cuda = false,
   useDevice = 1,
   -- Batch size
   batchSize = 192
}

opt.trainPath = opt.trainPath or paths.concat(opt.dataPath, 'ILSVRC2012_img_train')
opt.validPath = opt.validPath or paths.concat(opt.dataPath, 'ILSVRC2012_img_val')
opt.metaPath = opt.metaPath or paths.concat(opt.dataPath, 'metadata')

print(opt)

model = nn.Sequential()
model:add(nn.SpatialConvolution(1,64,5,5,1,1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2))
model:add(nn.SpatialConvolution(64,128,5,5,1,1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2))
model:add(nn.Collapse(3))
model:add(nn.Linear(128*4*4, 11))
model:add(nn.LogSoftMax())
model:float()

mnist = dp.Mnist{input_preprocess = {dp.Standardize()}}
ds_train = mnist:trainSet()
ds_valid = mnist:validSet()

-- loss
loss = nn.ClassNLLCriterion()
loss:float()


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
   inputs = inputs:permute(1,4,2,3):float()
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
--preprocess = ds:normalizePPF()

-- Sample one epoch!
exp = TrainHelpers.ExperimentHelper{
   model = model,
   trainDataset = ds_train,
   batchSize = opt.batchSize,
   --preprocessFunc = preprocess,
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   dampening = opt.dampening,
   nesterov = opt.nesterov,
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   datasetMultithreadLoading = 0
}
exp:printEpochProgress{everyNBatches = 1}
exp:printAverageTrainLoss{everyNBatches = 10}
exp:trainForever()
