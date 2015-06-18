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