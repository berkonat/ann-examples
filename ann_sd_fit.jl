using Flux
using Flux: Chain, Dense, params, train!, glorot_uniform
using Flux.Tracker: Params, gradient, update!

f = x -> (1/(0.82+x))^12 -2 * (1/(.82+x))^6
#f = x -> sin.(x.*7.0)

nsamples = 200
XS = rand(nsamples)
YS = f.(XS)
#samples = zip(XS,YS)
#samples = [(x, f(x)) for x in XS]
#samples= [(XS, YS)]

xerr = range(0, 1, length=1000)
eval_nn(model, x) = model(x).data
maxfit_nn(model) = maximum(abs, f.(XS) - model(XS))

function train_nn(nnodes, η, samples, XX, YY)
    model = Chain(Dense(nnodes, nnodes, Flux.tanh),   # first hidden layer
                  Dense(nnodes, 1))      # output layer
    
    loss(x, y) = Flux.mse(model(x), y)

    θ = Flux.params(model)
    #println(θ)

    for i=1:1000
        #Flux.train!(loss, θ, samples, Descent(η))
        Flux.train!(loss, θ, repeated((samples,), 2_000), ADAM(0.05))
        if i%100==1
            @show loss(XX, YY)
            @show maxfit_nn(model)
        end
    end
    return model
end

η = 0.00001 # Learning rate (should be smaller corresponding to the complex regression times the # of nodes)
NN_nodes = 3:5
errs_nn = Float64[]

for N in NN_nodes
    XX = repeat(XS,1,N)
    XX = XX'
    samples = [([x for i=1:N], f(x)) for x in XS]
    @show N
    nn = train_nn(N,η,samples,XX,YS)
    #println(Flux.params(nn))
    #@show e = err_nn(nn)
    #push!(errs_nn, e)
end
@show errs_nn
