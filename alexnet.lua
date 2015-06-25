require 'dp'
require 'optim'
require 'cunn'
require 'TrainHelpers'
require 'cutorch'
require 'cudnn'
require 'inn'

------------ Model ------------

model = nn.Sequential()
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

loss = nn.ClassNLLCriterion()
loss:cuda()


------------ Dataset ------------

dataPath = paths.concat(dp.DATA_DIR, 'ImageNet')
ds_all = dp.ImageNet{
   train_path = paths.concat(dataPath, 'ILSVRC2012_img_train'),
   valid_path = paths.concat(dataPath, 'ILSVRC2012_img_val'),
   meta_path = paths.concat(dataPath, 'metadata'),
   verbose = true,
}
ds_train = ds_all:loadTrain()
ds_val = ds_all:loadValid()
preprocess = TrainHelpers.normalizePreprocessDataset(ds_train, 255)
sampler = dp.RandomSampler{
    batch_size = 128,
    ppf = preprocess
}
val_sampler = dp.RandomSampler{
    batch_size = 12,
    ppf = preprocess
}
ds_train:multithread(4)
ds_val:multithread(4)
sampler:async()
val_sampler:async()


------------ Actual Training ------------

weights, gradients = model:getParameters()

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

sgdState = {
   learningRate = 0.01,
   momentum     = 0.9,
   dampening    = 0,
   weightDecay  = 0.0005,
   nesterov     = true,
   epochCounter = 1,
   lossLog = {},
   accuracyLog = {}
}

TrainHelpers.trainForever(model, forwardBackwardBatch, weights, sgdState, sampler, ds_train, val_sampler, ds_val, "snapshots/alexnet")
