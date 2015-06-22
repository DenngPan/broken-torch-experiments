require 'dp'
require 'optim'

TrainHelpers = {}

-- clear the intermediate states in the model before saving to disk
-- this saves lots of disk space
function sanitize(net)
   local list = net:listModules()
   for _,val in ipairs(list) do
         for name,field in pairs(val) do
            if torch.type(field) == 'cdata' then val[name] = nil end
            if name == 'homeGradBuffers' then val[name] = nil end
            if name == 'input_gpu' then val[name] = {} end
            if name == 'input' then val[name] = {} end
            if name == 'finput' then val[name] = {} end
            if name == 'gradOutput_gpu' then val[name] = {} end
            if name == 'gradOutput' then val[name] = {} end
            if name == 'fgradOutput' then val[name] = {} end
            if name == 'gradInput_gpu' then val[name] = {} end
            if name == 'gradInput' then val[name] = {} end
            if name == 'fgradInput' then val[name] = {} end
            if (name == 'output' or name == 'gradInput') then
               val[name] = field.new()
            end
         end
   end
end

function inspect(model)
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

local ExperimentHelper = torch.class('TrainHelpers.ExperimentHelper')
function ExperimentHelper:__init(config)
   self.model = config.model
   self.trainDataset = config.trainDataset
   self.epochCounter = 0
   self.batchCounter = 0
   self.batchSize = config.batchSize
   self.totalNumSeenImages = 0
   self.currentEpochSeenImages = 0
   self.currentEpochSize = 0
   self.callbacks = {}
   self.lossLog = {}
   self.preprocessFunc = config.preprocessFunc

   self.sgdState = config.sgdState or {
      learningRate = config.learningRate,
      --learningRateDecay = 1e-7,
      weightDecay = config.weightDecay,
      momentum = config.momentum,
      dampening = config.dampening,
      nesterov = config.nesterov,
   }
   print("SGD state:", self.sgdState)

   self.sampler = config.sampler or dp.RandomSampler{
      batch_size = self.batchSize,
      ppf = self.preprocessFunc
   }
   if config.datasetMultithreadLoading > 0 then
      self.trainDataset:multithread(config.datasetMultithreadLoading)
      self.sampler:async()
   end

end
function ExperimentHelper:runEvery(config, func)
   if config.everyNBatches then
      self.callbacks[config.everyNBatches] = self.callbacks[config.everyNBatches] or {}
      table.insert(self.callbacks[config.everyNBatches], func)
   end
end
function ExperimentHelper:printEpochProgress(freq)
   self:runEvery(freq,
                 function()
                    xlua.progress(self.currentEpochSeenImages,
                                  self.currentEpochSize)
              end)
end
function ExperimentHelper:printAverageTrainLoss(freq, nBatchAverage)
   -- nBatchAverage = nBatchAverage or 10
   self:runEvery(freq,
                 function()
                     -- local loss = 0
                     -- local before,after = table.splice(self.lossLog, #self.lossLog-nBatchAverage, nBatchAverage)
                     -- for _, entry in ipairs(after) do
                     --     loss = loss + entry.loss
                     -- end
                     -- print("Average loss for batches "..(self.batchCounter-nBatchAverage).."--"..self.batchCounter..":", loss/#after)
                     print("Loss for batch "..self.batchCounter.." is", self.lossLog[#self.lossLog].loss)
                 end
   )

end
function ExperimentHelper:snapshotModel(config)
   self:runEvery(config,
                 function()
                    local filename = string.format(config.filename, self.totalNumSeenImages)
                    print("Saving experiment state to", filename)
                    sanitize(self.model)
                    torch.save(filename, {
                        model=self.model,
                        sgdState=self.sgdState,
                        lossLog=self.lossLog
                    })

                 end)
end
function ExperimentHelper:trainEpoch()
   self.epochCounter = self.epochCounter + 1
   print("---------- Epoch "..self.epochCounter.." ----------")
   local epoch_sampler = self.sampler:sampleEpoch(self.trainDataset)
   local batch
   local l
   local epochSize
   local new_w
   while true do
       batch, self.currentEpochSeenImages, self.currentEpochSize = epoch_sampler(batch)
       collectgarbage(); collectgarbage()
       if not batch then
          break -- Epoch done
       end
      local inputs = batch:inputs():input()
      local targets = batch:targets():input()
      new_w, l = optim.sgd(function()
                              return eval(batch:inputs():input(),
                                          batch:targets():input())
                           end, weights, sgdState)
      self.batchCounter = self.batchCounter + 1
      self.totalNumSeenImages = self.totalNumSeenImages + batch:targets():input():size(1)
      table.insert(self.lossLog, {loss=l[1], totalNumSeenImages=self.totalNumSeenImages})

      for frequency, funcs in pairs(self.callbacks) do
         if self.batchCounter % frequency == 0 then
             io.write("\027[K") -- clear line (useful for progress bar)
             for _,func in ipairs(funcs) do
                 func(self)
             end
         end
      end
   end
end
function ExperimentHelper:trainForever()
    while true do
        self:trainEpoch()
    end
end

function TrainHelpers.normalizePreprocessDataset(dataset)
   local meanstdCache = paths.concat(dataset._data_path[1], 'mean_img.th7')
   local mean, std
   if paths.filep(meanstdCache) then
      local meanstd = torch.load(meanstdCache)
      mean = meanstd.mean
      std = meanstd.std
      print('Loaded mean and std from cache.')
   else
      local tm = torch.Timer()
      local nSamples = 10000
      print('Estimating the mean,std images over '
            .. nSamples .. ' randomly sampled training images')

      mean = nil
      std = nil
      local batch
      for i=1,nSamples,100 do
         xlua.progress(i, nSamples)
         batch = dataset:sample(batch, 100)
         local input = batch:inputs():forward('bchw')
         if not mean then
            mean = input:mean(1)
            std = input:std(1)
         else
            mean:add(input:mean(1):expandAs(mean))
            std:add(input:std(1):expandAs(mean))
         end
      end
      print ""
      mean = mean*100 / nSamples
      std = std*100 / nSamples
      local cache = {mean=mean,std=std}
      torch.save(meanstdCache, cache)

      print('Time to estimate:', tm:time().real)
   end

   local function ppf(batch)
      local inputView = batch:inputs()
      local input = inputView:input()
      input:add(-mean:expandAs(input)):cdiv(std:expandAs(input))
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