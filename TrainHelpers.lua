require 'dp'
require 'xlua'
require 'optim'

TrainHelpers = {}

function TrainHelpers.inspect(model)
   local list = model:listModules()
   local fields = {}
   for i, module in ipairs(list) do
      print("Module "..i.."------------")
      for n,val in pairs(module) do
         local str
         if torch.isTensor(val) then
            str = torch.typename(val).." of size "..val:numel()
         else
            str = tostring(val)
         end
         table.insert(fields,n)
         print("    "..n..": "..str)
      end
   end

   print("Unique fields:")
   print(_.uniq(fields))
end

-- local ExperimentHelper = torch.class('TrainHelpers.ExperimentHelper')
-- function ExperimentHelper:__init(config)
--    self.model = config.model
--    self.trainDataset = config.trainDataset
--    self.epochCounter = 0
--    self.batchCounter = 0
--    self.batchSize = config.batchSize
--    self.totalNumSeenImages = 0
--    self.currentEpochSeenImages = 0
--    self.currentEpochSize = 0
--    self.callbacks = {}
--    self.lossLog = {}
--    self.preprocessFunc = config.preprocessFunc

--    self.sgdState = config.sgdState or {
--       learningRate = config.learningRate,
--       --learningRateDecay = 1e-7,
--       weightDecay = config.weightDecay,
--       momentum = config.momentum,
--       dampening = config.dampening,
--       nesterov = config.nesterov,
--    }
--    print("SGD state:", self.sgdState)

--    self.sampler = config.sampler or dp.RandomSampler{
--       batch_size = self.batchSize,
--       ppf = self.preprocessFunc
--    }
--    if config.datasetMultithreadLoading > 0 then
--       self.trainDataset:multithread(config.datasetMultithreadLoading)
--       self.sampler:async()
--    end

-- end
-- function ExperimentHelper:runEvery(config, func)
--    if config.everyNBatches then
--       self.callbacks[config.everyNBatches] = self.callbacks[config.everyNBatches] or {}
--       table.insert(self.callbacks[config.everyNBatches], func)
--    end
-- end
-- function ExperimentHelper:printEpochProgress(freq)
--    self:runEvery(freq,
--                  function()
--                     xlua.progress(self.currentEpochSeenImages,
--                                   self.currentEpochSize)
--               end)
-- end
-- function ExperimentHelper:printAverageTrainLoss(freq, nBatchAverage)
--    -- nBatchAverage = nBatchAverage or 10
--    self:runEvery(freq,
--                  function()
--                      -- local loss = 0
--                      -- local before,after = table.splice(self.lossLog, #self.lossLog-nBatchAverage, nBatchAverage)
--                      -- for _, entry in ipairs(after) do
--                      --     loss = loss + entry.loss
--                      -- end
--                      -- print("Average loss for batches "..(self.batchCounter-nBatchAverage).."--"..self.batchCounter..":", loss/#after)
--                      print("Loss for batch "..self.batchCounter.." is", self.lossLog[#self.lossLog].loss)
--                  end
--    )

-- end
-- function ExperimentHelper:snapshotModel(config)
--    self:runEvery(config,
--                  function()
--                     local filename = string.format(config.filename, self.totalNumSeenImages)
--                     print("Saving experiment state to", filename)
--                     sanitize(self.model)
--                     torch.save(filename, {
--                         model=self.model,
--                         sgdState=self.sgdState,
--                         lossLog=self.lossLog
--                     })

--                  end)
-- end
-- function ExperimentHelper:trainEpoch(eval, weights)
--    self.epochCounter = self.epochCounter + 1
--    print("---------- Epoch "..self.epochCounter.." ----------")
--    local epoch_sampler = self.sampler:sampleEpoch(self.trainDataset)
--    local batch
--    local l
--    local epochSize
--    local new_w
--    while true do
--        batch, self.currentEpochSeenImages, self.currentEpochSize = epoch_sampler(batch)
--        collectgarbage(); collectgarbage()
--        if not batch then
--           break -- Epoch done
--        end
--       local inputs = batch:inputs():input()
--       local targets = batch:targets():input()
--       new_w, l = optim.sgd(function()
--                               return eval(batch:inputs():input(),
--                                           batch:targets():input())
--                            end, weights, self.sgdState)
--       self.batchCounter = self.batchCounter + 1
--       self.totalNumSeenImages = self.totalNumSeenImages + batch:targets():input():size(1)
--       table.insert(self.lossLog, {loss=l[1], totalNumSeenImages=self.totalNumSeenImages})

--       for frequency, funcs in pairs(self.callbacks) do
--          if self.batchCounter % frequency == 0 then
--              io.write("\027[K") -- clear line (useful for progress bar)
--              for _,func in ipairs(funcs) do
--                  func(self)
--              end
--          end
--       end
--    end
-- end
-- function ExperimentHelper:trainForever(eval, weights)
--     while true do
--         self:trainEpoch(eval, weights)
--     end
-- end

function TrainHelpers.normalizePreprocessDataset(dataset, rawScale)
   -- Returns a function that
   if not rawScale then
      -- Important! ImageNet models like to have input scaled to
      -- 0-255, not 0-1! After mean subtraction, this results in
      -- inputs with range approximately [-128,128]. This matches with
      -- Caffe's defaults, and all of Caffe's reference models do
      -- something like this.
      rawScale = 255
   end
   local meanstdCache = paths.concat(dataset._data_path[1], 'mean_img.th7')
   local mean, std
   if paths.filep(meanstdCache) then
      local meanstd = torch.load(meanstdCache)
      mean = meanstd.mean
      -- std = meanstd.std
      print('Loaded mean and std from cache.')
   else
      local tm = torch.Timer()
      local nSamples = 10000
      print('Estimating the mean images over '
            .. nSamples .. ' randomly sampled training images')

      mean = nil
      -- std = nil
      local batch
      for i=1,nSamples,100 do
         xlua.progress(i, nSamples)
         batch = dataset:sample(batch, 100)
         local input = batch:inputs():forward('bchw')
         if not mean then
            mean = input:mean(1)
            -- std = input:std()
         else
            mean:add(input:mean(1):expandAs(mean))
            -- std = std + input:std()
         end
      end
      print ""
      mean = mean*100 / nSamples
      --std = std*100 / nSamples
      local cache = {mean=mean} --,std=std}
      torch.save(meanstdCache, cache)

      print('Time to estimate:', tm:time().real)
   end

   local function ppf(batch)
       local inputView = batch:inputs()
       local input = inputView:input()
       input:add(-mean:expandAs(input)):mul(rawScale)
       return batch
   end

   if dataset._verbose then
      -- just check if mean/std look good now
      local batch = dataset:sample(100)
      ppf(batch)
      local input = batch:inputs():input()
      print('Stats of 100 randomly sampled images after normalizing. '..
            'Mean: ' .. input:mean().. ' Std: ' .. input:std())
   end
   return ppf
end


function TrainHelpers.evaluateModel(model, epoch, cuda, useTenCrops)
   -- Evaluate the model and report accuracy
   print "Evaluating model..."
   local total = 0
   local correct1 = 0
   local correct5 = 0
   local batch, imagesSeen, epochSize
   model:evaluate() -- sets flag (for dropout, etc)
   while true do -- Each batch
      collectgarbage(); collectgarbage()
      batch,imagesSeen,epochSize = epoch(batch)
      if not batch then
         break
      end
      local inputs = batch:inputs():input()
      local targets = batch:targets():input():long()
      if cuda then
         inputs = inputs:cuda()
      end
      local outputs = model:forward(inputs):float()
      -- Take the average of all ten crops
      if useTenCrops then
          assert(outputs:size(1) % 10 == 0, "Uh oh -- number of crops is not divisible by 10.")
          outputs = outputs:view(outputs:size(1)/10, 10, outputs:size(2)):mean(2):select(2,1)
          targets = targets:view(targets:size(1)/10, 10):select(2,1):clone()
      end
      local _, indices = torch.sort(outputs, 2, true)
      -- indices has shape (batchSize, nClasses)
      local top1 = indices:select(2, 1)
      local top5 = indices:narrow(2, 1,5)
      correct1 = correct1 + torch.eq(top1, targets):sum()
      correct5 = correct5 + torch.eq(top5,
                                     targets:view(-1, 1):expandAs(top5)
                                  ):sum()
      total = total + indices:size(1)
      xlua.progress(imagesSeen, epochSize)
   end
   print("\n")
   print("Top 1 accuracy: "..(correct1 / total))
   print("Top 5 accuracy: "..(correct5 / total))
   return {correct1=correct1, correct5=correct5}
end


function TrainHelpers.trainForever(model, forwardBackwardBatch, weights, sgdState, sampler, ds_train, val_sampler, ds_val, filename, useCuda, useTenCrops, epochDropCount)
   if useCuda == nil then useCuda = true end
   if useTenCrops == nil then useTenCrops = true end
   if epochDropCount == nil then epochDropCount = 20 end
   local modelTag = torch.random()
   while true do -- Each epoch
      sgdState.epochCounter = sgdState.epochCounter + 1
      local epoch = sampler:sampleEpoch(ds_train)
      local batch,imagesSeen,epochSize
      print("---------- Epoch "..sgdState.epochCounter.." ----------")
      while true do -- Each batch
          collectgarbage(); collectgarbage()
          batch,imagesSeen,epochSize = epoch(batch)
          if not batch then
              break
          end
          -- Run forward and backward pass on inputs and labels
          model:training()
          local loss_val, gradients = forwardBackwardBatch(
              batch:inputs():input():cuda(),
              batch:targets():input():cuda()
          )
          -- SGD step: modifies weights in-place
          optim.sgd(function() return loss_val, gradients end,
                      weights,
                      sgdState)
          -- Display progress and loss
          xlua.progress(imagesSeen, epochSize)
          if sgdState.evalCounter % 20 == 0 then
              print("\027[KBatch "..sgdState.evalCounter.." loss:", loss_val)
          end
          table.insert(sgdState.lossLog, loss_val)
      end
      -- Epoch completed!
      -- Every so often, decrease learning rate
      if sgdState.epochCounter % epochDropCount == 0 then
          sgdState.learningRate = sgdState.learningRate * 0.1
          print("Dropped learning rate to", sgdState.learningRate)
      end
      -- Snapshot model.
      if filename then
          torch.save(filename.."-latest.t7.tmp", { --epoch"..sgdState.epochCounter..".t7", {
              -- I would love to be able to just save the weights and be done
              -- with it, but I can't because Batch Norm layers require extra
              -- fields.
              model = model,
              sgdState = sgdState,
          })
          os.rename(filename.."-latest.t7.tmp", filename.."-latest-"..modelTag..".t7") -- POSIX guarantees automicity
      end
      -- Evaluate model
      table.insert(sgdState.accuracyLog, {
          epochCounter = sgdState.epochCounter,
          results = TrainHelpers.evaluateModel(model, val_sampler:sampleEpoch(ds_val), useCuda, useTenCrops)
      })
   end
end
