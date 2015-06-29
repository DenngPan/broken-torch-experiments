require 'dp'
require 'optim'
require 'nn'
require 'cunn'
require 'TrainHelpers'
require 'cutorch'
require 'cudnn'
require 'inn'
require 'ETL-Kanji-set'

------------- Model -------------

-- Just like 'KanjiNet-allconv.lua', but this time I don't bungle all the
-- strides.
--
-- Based in spirit on the "All Convolutional Network". Several parts
-- of the network:
-- * It's all convolutional: no fully connected layers, no max pooling
-- * Global average pooling at the end, for translation invariance
-- * Parts are similar to network-in-network
-- * Batch normalization everywhere!

model = nn.Sequential()
model:add(nn.Narrow(2,1,1)) -- Convert RGB to grayscale
model:add(cudnn.SpatialConvolution(1, 64, 7,7, 2,2))
model:add(nn.SpatialBatchNormalization(64, 1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(64, 64, 1,1, 1,1))
model:add(nn.SpatialBatchNormalization(64, 1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(64, 64, 3,3, 2,2))
model:add(nn.SpatialBatchNormalization(64, 1e-3))
model:add(cudnn.ReLU(true))
-- 14x14
model:add(cudnn.SpatialConvolution(64, 256, 3,3, 1,1))
model:add(nn.SpatialBatchNormalization(256, 1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(256, 256, 1,1, 1,1))
model:add(nn.SpatialBatchNormalization(256, 1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(256, 256, 3,3, 2,2))
model:add(nn.SpatialBatchNormalization(256, 1e-3))
model:add(cudnn.ReLU(true))
-- 5x5
model:add(cudnn.SpatialConvolution(256, 957, 3,3, 1,1))
model:add(nn.SpatialAveragePooling(3,3, 1,1))
model:add(nn.View(957))
model:add(nn.LogSoftMax())

cutorch.setDevice(1)
model:cuda()

loss = nn.ClassNLLCriterion()
loss:cuda()

------------- Dataset -------------
trainSet, validSet = ETLKanjiSet.load_8g(
   paths.concat(dp.DATA_DIR, 'ETL-kanji-datasets')
)
preprocess = TrainHelpers.normalizePreprocessDataset(trainSet, 255)

sampler = dp.RandomSampler{
    batch_size = 64,
    ppf = preprocess
}
validSampler = dp.RandomSampler{
    batch_size = 64,
    ppf = preprocess
}
trainSet:multithread(4)
validSet:multithread(4)
sampler:async()
validSampler:async()

------------ Actual Training ------------

weights, gradients = model:getParameters()

function forwardBackwardBatch(inputs, targets)
   gradients:zero()
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
   epochCounter = 0,
   lossLog = {},
   accuracyLog = {}
}

TrainHelpers.trainForever(
   model,
   forwardBackwardBatch,
   weights,
   sgdState,
   sampler,
   trainSet,
   validSampler,
   validSet,
   "snapshots-kanjinet-8g/KanjiNet-allconv-small-nodropout-20150626",
   true, -- useCuda
   false, -- useTenCrops
   5 -- epochDropCount
)
