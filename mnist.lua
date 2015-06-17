require 'dp'
require 'optim'

opt = {
   -- Path to ImageNet
   dataPath = paths.concat(dp.DATA_DIR, 'ImageNet'),
   nThread = 2,
   -- overwrite cache?
   overwrite = false,
   -- Learning schedule parameters
   learningRate = 0.01,
   schedule = {[1]=1e-2,[19]=5e-3,[30]=1e-3,[44]=5e-4,[53]=1e-4},
   -- max norm each layers output neuron weights
   maxOutNorm = -1,
   -- Weight decay
   weightDecay = 5e-4,
   momentum = 0.9,
   -- CUDA devices
   cuda = true,
   useDevice = 1,
   -- Epoch sizes
   trainEpochSize = 10,
   batchSize = 16,
   machEpoch = 100,
   maxTries = 30,
   -- Accumulate gradients in place?
   accUpdate = false,
   -- Reporting
   verbose = true,
   progress = true,

}

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

ds, ds_test = dp.Mnist{input_preprocess = {dp.Standardize()}}:loadTrainValid()

-- loss
criterion = nn.ClassNLLCriterion()
criterion:float()
--confusion = optim.ConfusionMatrix(_.keys(ds:classes()))


sgd_state = {
   learningRate = 0.01,
   --learningRateDecay = 1e-7,
   --weightDecay = 1e-5,
   momentum = 0.9,
   dampening = 0,
   nesterov = true,
}

weights,gradients = model:getParameters()

function fx()
   model:training()
   batch = ds:batch(512)
   inputs = batch:inputs():input():permute(1,4,2,3)
   targets = batch:targets():input()
   gradients:zero() -- should i do this instead...?

   local y = model:forward(inputs)
   local loss = criterion:forward(y, targets)
   local df_dw = criterion:backward(y, targets)
   model:backward(inputs, df_dw)
   return loss, gradients
end


function eval_test()
   local batch = ds_test:batch(1000)
   local inputs = batch:inputs():input():permute(1,4,2,3)
   local targets = batch:targets():input()
   local confusion = optim.ConfusionMatrix(_.range(10))
   confusion:batchAdd(model:forward(inputs), targets)
   print(confusion)
end


function train_more()
    for i = 1,10 do
       print("Gradient sig:",gradients:clone():abs():sum())
       print("Weights sig:",weights:clone():abs():sum())
       new_w, l = optim.sgd(fx, weights, sgd_state)
       print("Now, weights sig:",weights:clone():abs():sum())
       print("New weights sig:",new_w:clone():abs():sum())
       --l,g = fx()
       --weights:add(gradients * -0.01)
       print(l)
       print(sgd_state)
    end
end
