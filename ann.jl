
using Flux
using Base.Iterators: repeated

##
# approximate this funcgtion on [0, 1]
# it is a small section of  the Lennard-Jones potential
f = x -> (1/(0.82+x))^12 -2 * (1/(.82+x))^6
# massage it so Flux can use it
trueFun(u) = [f(u[1])]

# Here we are taking measurements from the true model
# In real applications, we would just be given this
# data rather than generate it ourselves
nsamples = 200
xs = rand(nsamples)
samples = [ [x] for x in xs ]
y = trueFun.(samples)

xerr = range(0, 1, length=1000)
eval_nn(nn, x) = nn([x])[1].data
err_nn(nn) = maximum(abs, f.(xerr) - eval_nn.(Ref(nn), xerr))
maxfit_nn(nn) = maximum(abs, f.(xs) - eval_nn.(Ref(nn), xs))

function train_nn(nnodes)
    # This defines an ANN with an input layer, one hidden and an output layer
    nn = Chain(x -> x,             # input layer
               Dense(1, nnodes, Flux.sigmoid),   # first hidden layer
               Dense(nnodes, 1))        # output layer

    # extract the parameters:
    # the weights w^{[n]} and shifts b^{[n]}
    ps = Flux.params(nn)

    # errors at individual sample points
    sqerrors() = [ z[1]^2 for z in y - Flux.data(nn.(samples))]
    # loss functional => Least squares
    losssq() = sum(sqerrors())/length(samples)
    # for plotting the errors
    errcols() = [sqrt(e.data) for e in sqerrors()]

    cb() = nothing

    ## and now we can watch the training
    best_nn = deepcopy(nn)
    best_fit = 1e30

    for n = 1:nnodes, _ = 1:4
        Flux.train!(losssq, ps, repeated((), 2_000), ADAM(0.05), cb = cb)
        @show maxfit_nn(nn)
        if maxfit_nn(nn) < best_fit
            best_nn = deepcopy(nn)
        end
    end
    return nn
end


NN_nn = 3:5
errs_nn = Float64[]

for N in NN_nn
    @show N
    nn = train_nn(N)
    @show e = err_nn(nn)
    push!(errs_nn, e)
end
@show errs_nn


# ## Now produce plots of the two functions
# nn10 = train_nn(10)
# nn20 = train_nn(20)
# xp = range(0, 1, length = 300)
# y10 = eval_nn.(Ref(nn10), xp)
# y20 = eval_nn.(Ref(nn20), xp)
# plot(xp, f.(xp))
# plot!(xp, y10)
# plot!(xp, y20)
