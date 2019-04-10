require "nn"
require "optim"

-- Dataset definition for Torch with size()
dataset={}   

function dataset:size() 
  return #dataset
end

-- Target function definition
function true_fun(x)
  -- inputs [0,1]
  -- return math.sin(x*2*math.pi)
  return (1/(0.82+x))^12 -2 * (1/(.82+x))^6
end

-- Generating training dataset
function gen_traindata(npoints) 
  for i=1,npoints do 
    xval = math.random()
    yval = true_fun(xval)
    table.insert(dataset,{torch.Tensor({xval}),torch.Tensor({yval})})
  end
  print (npoints .. ' data points added to the data set')
end

function network_preds(npoints)  -- generate data to test NN's predictions
  outfile=io.open('output','w')
  xvals = {}
  for i=1,npoints do
    xval=(npoints-i)/(npoints-1)
    yreal=true_fun(xval)
    xinput=torch.Tensor({xval})
    ypred=mlp:forward(xinput)
    print(xval,yreal,ypred[1])
    outfile:write(xval .. ' ', yreal .. ' ', ypred[1], "\n")
  end
  outfile:close()
end  

function prediction_error(npoints)  -- generate data to test NN's predictions
  err = 0.0
  for i=1,npoints do
    xval=(npoints-i)/(npoints-1)
    yreal=true_fun(xval)
    xinput=torch.Tensor({xval})
    ypred=mlp:forward(xinput)
    err = err + math.abs(ypred[1]-yreal)
  end
  print('Mean Error:' .. err/npoints)
end  

function gen_network(ninputs,noutputs,nhidden)
  mlp=nn.Sequential()
  mlp:add(nn.Linear(ninputs,nhidden))
  mlp:add(nn.Sigmoid())
  mlp:add(nn.Linear(nhidden,noutputs))
  return mlp
end

-- Generate dataset
data_size = 200
gen_traindata(data_size)
print(dataset:size())
inputdata = torch.DoubleTensor(dataset:size(),1)
outputdata = torch.DoubleTensor(dataset:size(),1)
for i = 1, dataset:size() do
   inputdata[i]=dataset[i][1]
   outputdata[i]=dataset[i][2]
end

maxIteration = 2000
maxTrainSteps = 100

for nnodes = 3, 6 do
  -- Creating ANN
  ninputs=1;noutputs=1;hidden=nnodes
  mlp = gen_network(ninputs,noutputs,hidden)
  params, gradParams = mlp:getParameters()
  print("Testing with #nodes:" .. nnodes)

  -- Optimization Settings
  local optimState = {learningRate = 0.01}
  criterion=nn.MSECriterion()

  for ntrain = 1, maxTrainSteps do
    -- Training
    for epoch = 1, maxIteration do
      function feval(params)
        gradParams:zero()

        local outputs = mlp:forward(inputdata)
        local loss = criterion:forward(outputs, outputdata)
        local dloss_doutputs = criterion:backward(outputs, outputdata)
        mlp:backward(inputdata, dloss_doutputs)

        return loss, gradParams
      end
      optim.adam(feval, params, optimState)
    end
    -- Predictions
    if ntrain%50 == 0 then
      prediction_error(1000)
    end
  end
end

-- network_preds(1000)

