require 'dp'

ETLKanjiSet = {}


function ETLKanjiSet.load_8g(path)
   local trainSet = dp.ImageClassSet{
      data_path = path.."/data-8g-train/",
      load_size = {3, 0,0}, -- Not used
      sample_size = {3, 64,64},
      sample_func = ETLKanjiSet.sampleTrain,
      which_set = 'train',
   }

   return trainSet
end

function ETLKanjiSet.sampleTrain(self, dst, path)
   -- Load the image into 'path' while applying the following
   -- transformations:
   -- * Random rotation, +- 5 degrees
   -- * Random scale, 0.9 -- 1.2
   -- * Random crops of 95% of the image size.
   if not path then
      path = dst
      dst = torch.FloatTensor()
   end
   local input = gm.Image():load(path, 128,127)
   -- do random rotation
   input:rotate(--math.max(-5, math.min(5, (torch.randn(1)[1] * 3))),
                torch.uniform(-5,5),
                255,255,255)
   -- do random scale
   input:size(nil, math.ceil(torch.uniform(64, 70)))
   -- do random crop
   local iW, iH = input:size()
   local oW = 64
   local oH = 64
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local out = input:crop(oW, oH, w1, h1)
   -- convert RGB to grayscale tensor
   out = out:toTensor('float','RGB','DHW', true)
   dst:resizeAs(out):copy(out)
   return dst
end
