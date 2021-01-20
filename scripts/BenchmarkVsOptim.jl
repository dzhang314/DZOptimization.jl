using Printf
using Random: seed!
using Statistics: median

using BenchmarkTools
using Optim

using DZOptimization
using DZOptimization.ExampleFunctions:
    rosenbrock_objective, rosenbrock_gradient!,
    riesz_energy, riesz_gradient!


function run_dzopt_benchmark(Optimizer, name, f, g!,
                             initial_point, initial_step_size)

    membench = @benchmark begin
        while !opt.has_converged[]
            step!(opt)
        end
    end samples=1 evals=1 setup=(
        opt = $Optimizer($f, $g!, $initial_point, $initial_step_size))
    @assert membench.memory == 0
    @assert membench.allocs == 0

    result = @benchmark begin
        opt = $Optimizer($f, $g!, $initial_point, $initial_step_size)
        while !opt.has_converged[]
            step!(opt)
        end
    end

    final = DZOptimization.run_and_test!(Optimizer(
        f, g!, initial_point, initial_step_size))

    @printf("%s on %s: minimum %.16e in %5d iterations; %8d nanoseconds (%8d bytes; %6d allocs)\n",
            rpad(string(Optimizer), 25), rpad(name, 10),
            final.current_objective_value[],
            final.iteration_count[],
            median(result.times), result.memory, result.allocs)
end


function run_optim_benchmark(Optimizer, name, f, g!, initial_point)

    result = @benchmark optimize($f, $g!, $initial_point,
                                 method=$Optimizer(), iterations=100_000)

    optimum = optimize(f, g!, initial_point,
                       method=Optimizer(), iterations=100_000)

    @printf("%s on %s: minimum %.16e in %5d iterations; %8d nanoseconds (%8d bytes; %6d allocs)\n",
            rpad(string(Optimizer), 25), rpad(name, 10),
            optimum.minimum,
            optimum.iterations,
            median(result.times), result.memory, result.allocs)

end


function run_comparative_benchmark(DZType, OptimType)

    run_dzopt_benchmark(DZType, "rosenbrock",
        rosenbrock_objective, rosenbrock_gradient!, [0.0, 0.0], 0.01)

    run_optim_benchmark(OptimType, "rosenbrock",
        rosenbrock_objective, rosenbrock_gradient!, [0.0, 0.0])

    seed!(0)
    points = 2.0 * rand(3, 10) .- 1.0

    run_dzopt_benchmark(DZType, "riesz",
        L2RegularizationWrapper(riesz_energy, 1.0),
        L2GradientWrapper(riesz_gradient!, 1.0),
        points, 0.001)

    run_optim_benchmark(OptimType, "riesz",
        L2RegularizationWrapper(riesz_energy, 1.0),
        L2GradientWrapper(riesz_gradient!, 1.0),
        points)

end


function main()
    run_comparative_benchmark(GradientDescentOptimizer, GradientDescent)
    run_comparative_benchmark(BFGSOptimizer, BFGS)
end


main()
