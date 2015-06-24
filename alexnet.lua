require 'dp'
require 'optim'
require 'cunn'
require 'TrainHelpers'
require 'cutorch'
require 'loadcaffe'
require 'cudnn'

------------ Model ------------

local model = nn.Sequential()
model:add(cudnn.SpatialConvolution(3,96,11,11,4,4,2,2))
    model.modules[#model.modules].weight:normal(0, 0.01)
    model.modules[#model.modules].bias:fill(0)
model:add(cudnn.ReLU())
model:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 1))
model:add(nn.SpatialMaxPooling(3,3,2,2))
model:add(cudnn.SpatialConvolution(96,256,5,5,1,1,2,2))
    model.modules[#model.modules].weight:normal(0, 0.01)
    model.modules[#model.modules].bias:fill(0.1)
model:add(cudnn.ReLU())
model:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 1))
model:add(nn.SpatialMaxPooling(3,3,2,2))
model:add(cudnn.SpatialConvolution(256,384,3,3,1,1,1,1))
    model.modules[#model.modules].weight:normal(0, 0.01)
    model.modules[#model.modules].bias:fill(0)
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(384,384,3,3,1,1,1,1))
    model.modules[#model.modules].weight:normal(0, 0.01)
    model.modules[#model.modules].bias:fill(0.1)
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(384,256,3,3,1,1,1,1))
    model.modules[#model.modules].weight:normal(0, 0.01)
    model.modules[#model.modules].bias:fill(0.1)
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3,3,2,2))

model:add(nn.View(256*6*6))
model:add(nn.Linear(256*6*6, 4096))
    model.modules[#model.modules].weight:normal(0, 0.005)
    model.modules[#model.modules].bias:fill(0.1)
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 4096))
    model.modules[#model.modules].weight:normal(0, 0.005)
    model.modules[#model.modules].bias:fill(0.1)
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 1000))
    model.modules[#model.modules].weight:normal(0, 0.01)
    model.modules[#model.modules].bias:fill(0)
model:add(nn.LogSoftMax())

cutorch.setDevice(1)
model:cuda()

local loss = nn.ClassNLLCriterion()
loss:cuda()


------------ Dataset ------------
local dataPath = paths.concat(dp.DATA_DIR, 'ImageNet')
local ds_all = dp.ImageNet{
   train_path = paths.concat(dataPath, 'ILSVRC2012_img_train'),
   valid_path = paths.concat(dataPath, 'ILSVRC2012_img_val'),
   meta_path = paths.concat(dataPath, 'metadata'),
   verbose = true,
}
local ds_train = ds_all:loadTrain()
local preprocess = TrainHelpers.normalizePreprocessDataset(ds_train, 255)
local ds_val = ds_all:loadValid()
local sampler = dp.RandomSampler{
    batch_size = 128,
    ppf = preprocess
}
ds_train:multithread(4)
sampler:async()

------------ Actual Training ------------
function forwardBackwardBatch(inputs, targets)
   model:training()
   gradients:zero() -- should i do this instead...?
   local y = model:forward(inputs)
   local loss_val = loss:forward(y, targets)
   local df_dw = loss:backward(y, targets)
   model:backward(inputs, df_dw)

   -- TODO: Optionally modify gradients for separate layers here
   -- eg. model.modules[43].gradBias:mul(2)

   return loss_val, gradients
end


-- Set up dataset
local lossLog = {}
local epochCounter = 0
local sgdState = {
   learningRate = 0.01,
   momentum=0.9,
   dampening=0,
   weightDecay=0.0005,
   nesterov=true
}
while true; do -- eoch epoch
   local epoch = sampler:sampleEpoch(ds_train)
   local batch,imagesSeen,epochSize
   while true; do -- each batch
      batch,imagesSeen,epochSize = epoch(batch)
      if not batch then
         break
      end
      -- Run forward and backward pass on inputs and labels
      local loss_val, gradients = forwardBackwardBatch(
         batch:inputs():input():cuda(),
         batch:targets():input():cuda()
      )
      -- SGD step
      optim.sgd(function() return loss, gradients end,
                weights,
                sgdState)
   end
   -- Epoch completed!
   epochCounter = epochCounter + 1
end