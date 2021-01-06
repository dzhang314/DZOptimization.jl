# DZOptimization.jl

#### Copyright © 2019-2021 by David K. Zhang. Released under the [MIT License](https://github.com/dzhang314/MultiFloats.jl/blob/master/LICENSE).

**DZOptimization.jl** is a Julia package for smooth nonlinear optimization that emphasizes performance, flexibility, and memory efficiency. In basic usage examples (see below), **DZOptimization.jl** has 6x less overhead and uses 10x less memory than [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

Unlike traditional optimization libraries which only provide a black-box `optimize` function (e.g., [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) and [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl)), **DZOptimization.jl** gives you full control of the optimization loop. This allows you to:

* interactively monitor the progress of an optimizer,
* interleave nonlinear optimization with other tasks,
* save/load data in the middle of optimization,
* run multiple optimizers in parallel, and
* terminate optimization whenever you want (as opposed to a [predetermined list of convergence criteria](https://github.com/JuliaOpt/NLopt.jl#using-with-mathoptinterface)).

**DZOptimization.jl** is designed to minimize overhead. It uses static data structures and in-place algorithms to ensure that memory is **never** dynamically allocated (outside of optimizer constructors). This makes **DZOptimization.jl** especially suitable for both small-scale optimization problems, since repeatedly allocating small vectors is wasteful, and large-scale optimization problems, since memory usage will never shoot up unexpectedly.


## Usage Example

The following example illustrates the use of `DZOptimization.BFGSOptimizer` to minimize the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function), starting at a random initial point.

```
using DZOptimization

rosenbrock_objective(x::Vector) =
    (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

function rosenbrock_gradient!(g::Vector, x::Vector)
    g[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    g[2] = 200 * (x[2] - x[1]^2)
end

opt = BFGSOptimizer(rosenbrock_objective,
                    rosenbrock_gradient!,
                    rand(2), # starting point
                    1.0)     # initial step size

while !opt.has_converged[]
    println(opt.current_objective_value[], '\t', opt.current_point)
    step!(opt)
end
```


## Benchmarks

Compared to [Optim.jl](http://julianlsolvers.github.io/Optim.jl/stable/), the BFGS implementation in **DZOptimization.jl** is 6x faster and uses 10x less memory to minimize the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function).

```
using BenchmarkTools
@benchmark begin
    opt = BFGSOptimizer(rosenbrock_objective,
                        rosenbrock_gradient!,
                        rand(2), 1.0)
    while !opt.has_converged[]; step!(opt); end
end

# BenchmarkTools.Trial: 
#   memory estimate:  1.14 KiB
#   allocs estimate:  12
#   --------------
#   minimum time:     2.800 μs (0.00% GC)
#   median time:      5.563 μs (0.00% GC)
#   mean time:        5.374 μs (0.64% GC)
#   maximum time:     182.925 μs (95.70% GC)
```

```
using Optim
@benchmark optimize(rosenbrock_objective,
                    rosenbrock_gradient!,
                    rand(2), BFGS())

# BenchmarkTools.Trial: 
#   memory estimate:  9.88 KiB
#   allocs estimate:  163
#   --------------
#   minimum time:     8.599 μs (0.00% GC)
#   median time:      30.300 μs (0.00% GC)
#   mean time:        31.826 μs (4.44% GC)
#   maximum time:     3.824 ms (98.29% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1
```
