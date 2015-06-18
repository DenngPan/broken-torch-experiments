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
   batchSize = 32
}

opt.trainPath = opt.trainPath or paths.concat(opt.dataPath, 'ILSVRC2012_img_train')
opt.validPath = opt.validPath or paths.concat(opt.dataPath, 'ILSVRC2012_img_val')
opt.metaPath = opt.metaPath or paths.concat(opt.dataPath, 'metadata')

print(opt)


--[[Model]]--

function buildInceptionLayer(inCh, n1ch, n3reduce, n3ch, n5reduce, n5ch, poolproj)
   local concat = nn.DepthConcat(2)
   local branch1x1 = nn.Sequential()
     branch1x1:add(nn.SpatialConvolution(inCh,n1ch,1,1))
     branch1x1:add(nn.ReLU(true))
   local branch3x3 = nn.Sequential()
     branch3x3:add(nn.SpatialConvolution(inCh,n3reduce,1,1))
     --branch3x3:add(nn.SpatialBatchNormalization(n3reduce))
     branch3x3:add(nn.ReLU(true))
     branch3x3:add(nn.SpatialConvolution(n3reduce,n3ch,3,3))
     branch3x3:add(nn.ReLU(true))
   local branch5x5 = nn.Sequential()
     branch5x5:add(nn.SpatialConvolution(inCh,n5reduce,1,1))
     --branch5x5:add(nn.SpatialBatchNormalization(n5reduce))
     branch5x5:add(nn.ReLU(true))
     branch5x5:add(nn.SpatialConvolution(n5reduce,n5ch,5,5))
     --branch5x5:add(nn.SpatialBatchNormalization(n5ch))
     branch5x5:add(nn.ReLU(true))
   local branch3x3MP = nn.Sequential()
     branch3x3MP:add(nn.SpatialMaxPooling(3,3, 2,2))
     branch3x3MP:add(nn.SpatialConvolution(inCh,poolproj,1,1))
     --branch3x3MP:add(nn.SpatialBatchNormalization(poolproj))
     branch3x3MP:add(nn.ReLU(true))
    concat:add(branch1x1)
    concat:add(branch3x3)
    concat:add(branch5x5)
    concat:add(branch3x3MP)
    return concat
   -- n.b.! since the convolutions have different sizes, many of these
   -- layers will be padded with zeros!
end

model = nn.Sequential()
model:add(nn.SpatialConvolution(3,64, 7,7,2,2, 3,3)) -- 224x224 -> 112x112
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 112x112 -> 56x56
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.SpatialConvolution(64,192, 3,3,1,1, 1,1))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 56x56 -> 28x28
model:add(nn.SpatialBatchNormalization(192))
---- Inception 3a
model:add(buildInceptionLayer(192, 64, 96, 128, 16, 32, 32))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(256))
---- Inception 3b
model:add(buildInceptionLayer(256, 128, 128, 192, 32, 96, 64))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 28x28 -> 14x14
model:add(nn.SpatialBatchNormalization(480))
---- Inception 4a
model:add(buildInceptionLayer(480, 192, 96, 208, 16, 48, 64))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(512))
---- Inception 4b
model:add(buildInceptionLayer(512, 160, 112, 224, 24, 64, 64))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(512))
---- Inception 4c
model:add(buildInceptionLayer(512, 128, 128, 256, 24, 64, 64))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(512))
---- Inception 4d
model:add(buildInceptionLayer(512, 112, 144, 288, 32, 64, 64))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(528))
---- Inception 4e
model:add(buildInceptionLayer(528, 256, 160, 320, 32, 128, 128))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(832))
---- Inception 5a
model:add(buildInceptionLayer(832, 256, 160, 320, 32, 128, 128))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(832))
---- Inception 5b
model:add(buildInceptionLayer(832, 384, 192, 384, 48, 128, 128))
model:add(nn.ReLU(true))
model:add(nn.SpatialAveragePooling(7,7,1,1))
model:add(nn.Dropout(0.4))
model:add(nn.Collapse(3))
model:add(nn.Linear(1024, 1000))
model:add(nn.LogSoftMax())
loss = nn.ClassNLLCriterion()

--[[data]]--
ds_all = dp.ImageNet{
   train_path=opt.trainPath, valid_path=opt.validPath, meta_path=opt.metaPath,
   verbose=true,
   cache_mode = opt.overwrite and 'overwrite' or nil
}
print "Got imagenet"
preprocess = ds_all:normalizePPF()
print "Got preprocessor"
ds_train = ds_all:loadTrain()
print "Got train"
ds_val = ds_all:loadValid()
print "Got valid"


-- CUDA-ize
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   model:cuda()
   loss:cuda()
end
print "Now on CUDA"

weights,gradients = model:getParameters() -- be sure to do this AFTER CUDA-izing it!
print "Have gradients..."

sgd_state = {
   learningRate = opt.learningRate,
   --learningRateDecay = 1e-7,
   --weightDecay = 1e-5,
   momentum = opt.momentum,
   dampening = opt.dampening,
   nesterov = opt.nesterov,
}

function fx()
   model:training()
   local batch = ds_train:sample(opt.batchSize)
   preprocess(batch)
   local inputs = batch:inputs():input()
   local targets = batch:targets():input()
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

print "Training...."
for i = 1,1000 do
    new_w, l = optim.sgd(fx, weights, sgd_state)
    print("Loss", l[1])
end
