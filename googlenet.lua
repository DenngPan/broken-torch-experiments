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
   batchSize = 128,
   machEpoch = 100,
   maxTries = 30,
   -- Accumulate gradients in place?
   accUpdate = false,
   -- Reporting
   verbose = true,
   progress = true,

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
model:add(nn.View( 1024))
model:add(nn.Linear(1024, 1000))
model:add(nn.LogSoftMax())

-- input = torch.randn(1,3, 224,224)
-- print(model:forward(input):size())

--model:add(nn.SpatialConvolution(64,192, 3,3,1,1))

-- local features = nn.Concat(2)
-- local fb1 = nn.Sequential() -- branch 1
-- fb1:add(nn.SpatialConvolution(3,48,11,11,4,4,2,2))       -- 224 -> 55
-- fb1:add(nn.ReLU())
-- if opt.LCN then
--    fb1:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 2))
-- end
-- fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
--
-- fb1:add(nn.SpatialConvolution(48,128,5,5,1,1,2,2))       --  27 -> 27
-- fb1:add(nn.ReLU())
-- if opt.LCN then
--    fb1:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 2))
-- end
-- fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
--
-- fb1:add(nn.SpatialConvolution(128,192,3,3,1,1,1,1))      --  13 ->  13
-- fb1:add(nn.ReLU())
--
-- fb1:add(nn.SpatialConvolution(192,192,3,3,1,1,1,1))      --  13 ->  13
-- fb1:add(nn.ReLU())
--
-- fb1:add(nn.SpatialConvolution(192,128,3,3,1,1,1,1))      --  13 ->  13
-- fb1:add(nn.ReLU())
--
-- fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
--
-- local fb2 = fb1:clone() -- branch 2
-- for k,v in ipairs(fb2:findModules('nn.SpatialConvolution')) do
--    v:reset() -- reset branch 2's weights
-- end
--
-- features:add(fb1)
-- features:add(fb2)
--
-- -- 1.3. Create Classifier (fully connected layers)
-- local classifier = nn.Sequential()
-- classifier:add(nn.Copy(nil, nil, true)) -- prevents a newContiguous in SpatialMaxPooling:backward()
-- classifier:add(nn.View(256*6*6))
-- classifier:add(nn.Dropout(0.5))
-- classifier:add(nn.Linear(256*6*6, 4096))
-- classifier:add(nn.Threshold(0, 1e-6))
--
-- classifier:add(nn.Dropout(0.5))
-- classifier:add(nn.Linear(4096, 4096))
-- classifier:add(nn.Threshold(0, 1e-6))
--
-- classifier:add(nn.Linear(4096, 1000))
-- classifier:add(nn.LogSoftMax())
--
-- -- 1.4. Combine 1.1 and 1.3 to produce final model
-- model = nn.Sequential()
-- model:add(nn.Convert(),1)
-- model:add(features)
-- model:add(classifier)
--

--[[data]]--
--- ds = dp.ImageNet{
---    train_path=opt.trainPath, valid_path=opt.validPath,
---    meta_path=opt.metaPath, verbose=opt.verbose,
---    cache_mode = opt.overwrite and 'overwrite' or nil
--- }

cub = require 'cub200-2011'
ds = cub.dataset

-- preprocessing function
ppf = ds:normalizePPF()

-- loss
criterion = nn.ClassNLLCriterion()
--confusion = optim.ConfusionMatrix(_.keys(ds:classes()))

weights,gradients = model:getParameters()

sgd_state = {
   learningRate = 1e-3,
   learningRateDecay = 0,
   weightDecay = 0,
   momentum = 0.9,
   dampening = 0.9,
   nesterov = false,
   learningRates = nil,
   weightDecays = nil,
}


function fx()
   local batch = ds:sample(opt.batchSize)
   ppf(batch)
   local inputs = batch:inputs():input()
   local targets = batch:targets():input()

   model:zeroGradParameters()
   -- model:syncParameters() -- only for DataParallelTable
   local timer = torch.Timer()
   local y = model:forward(inputs)
   local loss = criterion:forward(y, targets)
   local forward_time = timer:time().real

   timer = torch.Timer()
   local df_dw = criterion:backward(y, targets)
   model:backward(inputs, df_dw)
   local backward_time = timer:time().real

   print("Forward: ", xlua.formatTime(forward_time/opt.batchSize), " Backward: ", xlua.formatTime(backward_time/opt.batchSize))

   -- Normalize gradients and loss
   gradients.div(inputs:size(1))
   loss = loss / inputs:size(1)

   return loss, gradients
end


batch = ds:sample(10)
inputs = batch:inputs():input()

model:float()
criterion:float()


-- model:float()
-- print(model:forward(inputs))


-- --[[Propagators]]--
-- train = dp.Optimizer{
--    acc_update = opt.accUpdate,
--    loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
--    callback = function(model, report)
--                  print(report)
--                  --opt.learningRate = opt.schedule[report.epoch] or opt.learningRate
--                  --if opt.accUpdate then
--                  --   model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
--                  --else
--                  model:updateGradParameters(opt.momentum) -- affects gradParams
--                  model:weightDecay(opt.weightDecay) --affects gradParams
--                  model:updateParameters(opt.learningRate) -- affects params
--                  --end
--                  model:maxParamNorm(opt.maxOutNorm) -- affects params
--                  model:zeroGradParameters() -- affects gradParams
--    end,
--    feedback = dp.Confusion(),
--    sampler = dp.RandomSampler{
--       batch_size=opt.batchSize,
--       epoch_size=opt.trainEpochSize,
--       ppf=ppf
--    },
--    progress = opt.progress
-- }
-- valid = dp.Evaluator{
--    feedback = dp.TopCrop{n_top={1,5,10},n_crop=10,center=2},
--    sampler = dp.Sampler{
--       batch_size=math.round(opt.batchSize/10),
--       ppf=ppf
--    }
-- }
--
-- --[[multithreading]]--
-- if opt.nThread > 0 then
--    ds:multithread(opt.nThread)
--    train:sampler():async()
--    valid:sampler():async()
-- end
--
-- --[[Experiment]]--
-- xp = dp.Experiment{
--    model = model,
--    optimizer = train,
--    validator = valid,
--    observer = {
--       dp.FileLogger(),
--       dp.EarlyStopper{
--          error_report = {'validator','feedback','topcrop','all',5},
--          maximize = true,
--          max_epochs = opt.maxTries
--       }
--    },
--    random_seed = os.time(),
--    max_epoch = opt.maxEpoch
-- }
--
-- --[[GPU or CPU]]--
-- if opt.cuda then
--    require 'cutorch'
--    require 'cunn'
--    cutorch.setDevice(opt.useDevice)
--    xp:cuda()
-- end
--
-- print"Model :"
-- print(model)
--
-- xp:run(ds)