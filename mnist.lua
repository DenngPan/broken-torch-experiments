require 'dp'
require 'optim'
require 'TrainHelpers'
require 'cutorch'
require 'cunn'
require 'cudnn'

cutorch.setDevice(1)

model = nn.Sequential()
model:add(cudnn.SpatialConvolution(1,64,5,5,1,1))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2,2))
model:add(cudnn.SpatialConvolution(64,128,5,5,1,1))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2,2))
model:add(nn.View(128*4*4))
model:add(nn.Linear(128*4*4, 11))
model:add(nn.LogSoftMax())
model:cuda()

mnist = dp.Mnist{input_preprocess = {dp.Standardize()}}
ds_train = mnist:trainSet()
ds_valid = mnist:testSet()

loss = nn.ClassNLLCriterion()
loss:cuda()

sampler = dp.ShuffleSampler{
    batch_size = 128
}
val_sampler = dp.ShuffleSampler{
    batch_size = 12
}

weights,gradients = model:getParameters() 

function forwardBackwardBatch(inputs, targets)
   model:training()
   inputs = inputs:permute(1,4,2,3):float()
   inputs = inputs:cuda()
   targets = targets:cuda()
   gradients:zero() -- should i do this instead...?
   local y = model:forward(inputs)
   local loss_val = loss:forward(y, targets)
   local df_dw = loss:backward(y, targets)
   model:backward(inputs, df_dw)
   return loss_val, gradients
end

sgdState = {
   learningRate = 0.001,
   momentum     = 0.9,
   dampening    = 0,
   weightDecay  = 0.0005,
   nesterov     = true,
   epochCounter = 1,
   lossLog = {},
   accuracyLog = {}
}

TrainHelpers.trainForever(model, forwardBackwardBatch, weights, sgdState, sampler, ds_train, val_sampler, ds_valid)
-- Currently, this does not work because 'evaluateModel' assumes that we use
-- many crops.
