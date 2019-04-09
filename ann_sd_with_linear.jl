using Flux
using Flux: Chain, Dense, params, train!, glorot_uniform
using Flux.Tracker: Params, gradient, update!

f = x -> (1/(0.82+x))^12 -2 * (1/(.82+x))^6
#f = x -> sin.(x.*7.0)

nsamples = 200
XS = rand(nsamples)
YS = f.(XS)
samples = zip(XS,YS)
#samples = [(x, f(x)) for x in XS]
#samples= [(XS, YS)]

xerr = range(0, 1, length=1000)
eval_nn(model, x) = model(x).data
maxfit_nn(model) = maximum(abs, f.(XS) - model(XS)')

struct Linear{F,S,T}
  W::S
  b::T
  σ::F
end

Linear(W, b) = Linear(W, b, identity)

function Linear(in::Integer, out::Integer, σ = identity)
  return Linear(param(randn(out, in)), param(randn(out)), σ)
end

@Flux.treelike Linear

function (a::Linear)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    σ.(W * x' .+ b)
end

# Try to avoid hitting generic matmul in some simple cases
# Base's matmul is so slow that it's worth the extra conversion to hit BLAS
(a::Linear{<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::Linear{<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

function train_nn(nnodes, η)
    model = Chain(Linear(1, nnodes, Flux.tanh),   # first hidden layer
                  #Dense(nnodes, nnodes), # second hidden layer
                  Dense(nnodes, 1))      # output layer
    
    loss(x, y) = Flux.mse(model(x), y)

    θ = Flux.params(model)
    #println(θ)

    for i=1:5000
        #Flux.train!(loss, model_weights, samples, Descent(0.1))
        for j=1:length(XS)
            g = gradient(() -> loss([XS[j]], YS[j]), θ)
        #g = gradient(() -> loss(XS, YS), θ)
            for x in θ
                update!(x, -g[x]*η)
            end
        end
        if i%100==1
            @show loss(XS, YS)
            #println(size(model(XS)),size(YS))
            @show maxfit_nn(model)
        end
        if i%200==1
            η = η * 1.1
        end
    end
    return model
end

η = 0.000005 # Learning rate (should be smaller corresponding to the complex regression times the # of nodes)
NN_nodes = 5:5:15
errs_nn = Float64[]

for N in NN_nodes
    @show N
    nn = train_nn(N,η)
    println(Flux.params(nn))
    #@show e = err_nn(nn)
    #push!(errs_nn, e)
end
@show errs_nn
