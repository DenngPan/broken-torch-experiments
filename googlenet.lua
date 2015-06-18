require 'dp'
require 'optim'

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
   cuda = true,
   useDevice = 1,
   -- Batch size
   batchSize = 16
}

opt.trainPath = (opt.trainPath == '') and paths.concat(opt.dataPath, 'ILSVRC2012_img_train') or opt.trainPath
opt.validPath = (opt.validPath == '') and paths.concat(opt.dataPath, 'ILSVRC2012_img_val') or opt.validPath
opt.metaPath = (opt.metaPath == '') and paths.concat(opt.dataPath, 'metadata') or opt.metaPath


--[[Model]]--

function buildInceptionLayer(inCh, n1ch, n3reduce, n3ch, n5reduce, n5ch, poolproj)
   local concat = nn.DepthConcat(2)
   local branch1x1 = nn.Sequential()
     branch1x1:add(nn.SpatialConvolution(inCh,n1ch,1,1))
     branch1x1:add(nn.ReLU())
   local branch3x3 = nn.Sequential()
     branch3x3:add(nn.SpatialConvolution(inCh,n3reduce,1,1))
     branch3x3:add(nn.ReLU())
     branch3x3:add(nn.SpatialConvolution(n3reduce,n3ch,3,3))
     branch3x3:add(nn.ReLU())
   local branch5x5 = nn.Sequential()
     branch5x5:add(nn.SpatialConvolution(inCh,n5reduce,1,1))
     branch5x5:add(nn.ReLU())
     branch5x5:add(nn.SpatialConvolution(n5reduce,n5ch,5,5))
     branch5x5:add(nn.ReLU())
   local branch3x3MP = nn.Sequential()
     branch3x3MP:add(nn.SpatialMaxPooling(3,3, 2,2))
     branch3x3MP:add(nn.SpatialConvolution(inCh,poolproj,1,1))
     branch3x3MP:add(nn.ReLU())
    concat:add(branch1x1)
    concat:add(branch3x3)
    concat:add(branch5x5)
    concat:add(branch3x3MP)
    return concat
   -- n.b.! since the convolutions have different sizes, many of these
   -- layers will be padded with zeros!
end

model = nn.Sequential()
model:add(nn.SpatialConvolution(3,64, 7,7,2,2, 3)) -- 224x224 -> 112x112
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 112x112 -> 56x56
model:add(nn.SpatialConvolution(64,192, 3,3,1,1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 56x56 -> 28x28
---- Inception 3a
model:add(buildInceptionLayer(192, 64, 96, 128, 16, 32, 32))
---- Inception 3b
model:add(buildInceptionLayer(256, 128, 128, 192, 32, 96, 64))
model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 28x28 -> 14x14
---- Inception 4a
model:add(buildInceptionLayer(480, 192, 96, 208, 16, 48, 64))
---- Inception 4b
model:add(buildInceptionLayer(512, 160, 112, 224, 24, 64, 64))
---- Inception 4c
model:add(buildInceptionLayer(512, 128, 128, 256, 24, 64, 64))
---- Inception 4d
model:add(buildInceptionLayer(512, 112, 144, 288, 32, 64, 64))
---- Inception 4e
model:add(buildInceptionLayer(528, 256, 160, 320, 32, 128, 128))
model:add(nn.SpatialMaxPooling(2,2,2,2))
---- Inception 5a
model:add(buildInceptionLayer(832, 256, 160, 320, 32, 128, 128))
---- Inception 5b
model:add(buildInceptionLayer(832, 384, 192, 384, 48, 128, 128))
model:add(nn.SpatialAveragePooling(7,7,1,1))
model:add(nn.Dropout(0.4))
model:add(nn.Collapse(3))
model:add(nn.Linear(1024, 1000))
model:add(nn.LogSoftMax())

--[[data]]--
ds_all = dp.ImageNet{
   train_path=opt.trainPath, valid_path=opt.validPath,
   meta_path=opt.metaPath, verbose=opt.verbose,
   cache_mode = opt.overwrite and 'overwrite' or nil
}
preprocess = ds_all:normalizePPF()
ds_train = ds_all:loadTrain()
ds_val = ds_all:loadValid()

-- loss
criterion = nn.ClassNLLCriterion()
--confusion = optim.ConfusionMatrix(_.keys(ds:classes()))

-- CUDA-ize
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   model:cuda()
   criterion:cuda()
   ds_train:cuda()
   ds_valid:cuda()
end

weights,gradients = model:getParameters() -- be sure to do this AFTER CUDA-izing it!

sgd_state = {
   learningRate = opt.learningRate,
   --learningRateDecay = 1e-7,
   --weightDecay = 1e-5,
   momentum = 0.9,
   dampening = 0,
   nesterov = true,
}

function fx()
   model:training()
   local batch = ds_train:batch(opt.batchSize)
   local inputs = batch:inputs():input():permute(1,4,2,3)
   local targets = batch:targets():input()
   gradients:zero() -- should i do this instead...?
   local y = model:forward(inputs)
   local loss = criterion:forward(y, targets)
   local df_dw = criterion:backward(y, targets)
   model:backward(inputs, df_dw)
   return loss, gradients
end

print "Training...."
for i = 1,10 do
    new_w, l = optim.sgd(fx, weights, sgd_state)
    print("Loss", l[1])
end