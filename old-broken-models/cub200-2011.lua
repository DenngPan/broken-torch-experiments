require 'dp'
-- local CubSource, DataSource = torch.class("CubSource", "dp.DataSource")

-- function load_cub(config)
--    local args, image_folder, image_list_filename, class_list_filename = xlua.unpack(
--       {config},
--       "CubSource",
--       "CUB-200 2011 dataset source",
--       {arg="image_folder", type="string"},
--       {arg="image_list_filename", type="string"},
--       {arg="class_list_filename", type="string"}
--    )
--    print(image_folder)
--    print(image_list_filename)

--    -- Read image names
--    local image_name_map = {}
--    local idxes = {}
--    local target_map = {}
--    for line in io.lines(image_list_filename) do
--       local idx, image = unpack(string.split(line, " "))
--       table.insert(image_name_map, tonumber(idx), image_folder .. "/" .. image)
--       table.insert(idxes, tonumber(idx))
--    end
--    for line in io.lines(class_list_filename) do
--       local idx, class = unpack(string.split(line, " "))
--       table.insert(target_map, tonumber(idx), tonumber(class))
--    end

--    -- Collect images and targets.
--    local image_names = {}
--    local targets = {}
--    for _,i in ipairs(idxes) do
--       table.insert(image_names, image_name_map[i])
--       table.insert(targets, target_map[i])
--    end

--    return {image_names=image_names, targets=targets}
-- end


-- c = load_cub{image_folder = "/Users/michael/CUB_200_2011/images/",
--              image_list_filename = "/Users/michael/CUB_200_2011/images.txt",
--              class_list_filename = "/Users/michael/CUB_200_2011/image_class_labels.txt",
--          }


-- require 'image'
-- for i, name in ipairs(c.image_names) do
--    xx = image.load(name)
--    xx = nil
--    collectgarbage()
--    xlua.progress(i, #c.image_names)
-- end

require 'dp'
dataset = dp.ImageClassSet{
   data_path = "/Users/michael/CUB_200_2011/images/",
   load_size = {3, 256, 256},
   which_set = 'train',
   sample_size = {3, 224, 224},
   verbose = 1,
   sample_func = 'sampleTrain',
   sort_func = function(x,y)
      return tonumber(x:match('[0-9]+')) < tonumber(y:match('[0-9]+'))
   end,
   cache_mode = "writeonce"
}

function dataset:normalizePPF()
   local meanstdCache = paths.concat(self._data_path[1], 'meanstd.th7')
   local mean, std
   if paths.filep(meanstdCache) then
      local meanstd = torch.load(meanstdCache)
      mean = meanstd.mean
      std = meanstd.std
      if self._verbose then
         print('Loaded mean and std from cache.')
      end
   else
      local tm = torch.Timer()
      local trainSet = self
      local nSamples = 10000
      if self._verbose then
         print('Estimating the mean,std images over '
               .. nSamples .. ' randomly sampled training images')
      end

      mean = nil
      std = nil
      local batch
      for i=1,nSamples,100 do
         xlua.progress(i, nSamples)
         batch = trainSet:sample(batch, 100)
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

      if self._verbose then
         print('Time to estimate:', tm:time().real)
      end
   end

   -- if self._verbose then
   --    print('Mean: ', mean[1], mean[2], mean[3], 'Std:', std[1], std[2], std[3])
   -- end

   local function ppf(batch)
      local inputView = batch:inputs()
      local input = inputView:input()
      input:add(-mean:expandAs(input)):cdiv(std:expandAs(input))
      return batch
   end

   if self._verbose then
      -- just check if mean/std look good now
      local batch = self:sample(100)
      ppf(batch)
      local input = batch:inputs():input()
      print('Stats of 100 randomly sampled images after normalizing. '..
            'Mean: ' .. input:mean().. ' Std: ' .. input:std())
   end
   return ppf
end

return {dataset=dataset}