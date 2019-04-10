require "nn"

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
  print('Mean Error:' .. err/npoints .. "\n")
end  

function gen_network(ninputs,noutputs,nhidden)
  mlp=nn.Sequential()
  mlp:add(nn.Linear(ninputs,nhidden))
  mlp:add(nn.Sigmoid())
  mlp:add(nn.Linear(nhidden,noutputs))
  return mlp
end

-- Generate dataset
gen_traindata(100)
print(type(dataset))
print(dataset:size())

-- Creating ANN
ninputs=1;noutputs=1;hidden=3
mlp = gen_network(ninputs,noutputs,hidden)

-- Optimization Settings
criterion=nn.MSECriterion()
trainer=nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 2000

-- Training
trainer:train(dataset)

-- Predictions
-- network_preds(1000)
prediction_error(1000)

