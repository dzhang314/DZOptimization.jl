using Dates: format, now
using DZOptimization.SortingNetworks
using JLD2: load_object, save_object


const SAVE_INTERVAL_NS = UInt64(60_000_000_000)
const N = parse(Int, ARGS[2])


if ARGS[1] == "W"
    const CONDITION_CODE = 'W'
    const CONDITION_TYPE = WeaklyNormalizedCondition{2 * N,N}
elseif ARGS[1] == "I"
    const CONDITION_CODE = 'I'
    const CONDITION_TYPE = IncompletelyNormalizedCondition{2 * N,N,2}
elseif ARGS[1] == "J"
    const CONDITION_CODE = 'J'
    const CONDITION_TYPE = IncompletelyNormalizedCondition{2 * N,N,4}
elseif ARGS[1] == "K"
    const CONDITION_CODE = 'K'
    const CONDITION_TYPE = IncompletelyNormalizedCondition{2 * N,N,6}
elseif ARGS[1] == "S"
    const CONDITION_CODE = 'S'
    const CONDITION_TYPE = StronglyNormalizedCondition{2 * N,N}
elseif ARGS[1] == "C"
    const CONDITION_CODE = 'C'
    const CONDITION_TYPE = CompletelyNormalizedCondition{2 * N,N}
else
    error("Invalid condition code $(ARGS[1]) (must be W, I, J, K, S, or C)")
end


function main()

    gen = MultiFloatTestGenerator{2 * N,N,N}()
    cond = CONDITION_TYPE()
    opt = SortingNetworkOptimizer(gen, cond, N; pareto_radius=10)

    last_save_ns = time_ns()
    last_save_file = nothing
    filename_prefix = "Search$(CONDITION_CODE)$(N)"

    if length(ARGS) >= 3
        @assert startswith(ARGS[3], filename_prefix)
        @assert endswith(ARGS[3], ".jld2")
        opt = load_object(ARGS[3])
        @assert opt isa SortingNetworkOptimizer{2 * N,Float64,
            MultiFloatTestGenerator{2 * N,N,N},CONDITION_TYPE}
        assert_valid(opt)
    end

    start_time_ns = time_ns()

    while true

        step!(opt; verbose=true)
        println()
        flush(stdout)

        num_passing = length(opt.passing_networks)
        num_failing = length(opt.failing_networks)
        num_test_cases = length(opt.test_cases)
        println("$(num_passing) passing networks; ",
            "$(num_failing) failing networks; ",
            "$(num_test_cases) test cases.")
        if !isempty(opt.pareto_frontier)
            counts = Dict{Tuple{Int,Int},Int}()
            for (network, _) in opt.passing_networks
                point = fitness(opt, network)
                if haskey(counts, point)
                    counts[point] += 1
                else
                    counts[point] = 1
                end
            end
            println([point => counts[point]
                     for point in sort!(collect(opt.pareto_frontier))])
        end
        println()
        flush(stdout)

        if !isempty(opt.passing_networks)
            network = argmin(opt.passing_networks)
            old_num_tests = opt.passing_networks[network]
            point = fitness(opt, network)
            println("Retesting $point network.")
            opt(network; verbose=true)
            if haskey(opt.passing_networks, network)
                new_num_tests = opt.passing_networks[network]
                println("Increased number of tests from ",
                    "$old_num_tests to $new_num_tests.")
            end
            println()
            flush(stdout)
        end

        if time_ns() - last_save_ns >= SAVE_INTERVAL_NS
            disable_sigint() do
                timestamp = format(now(), "yyyy-mm-dd-HH-MM-SS")
                filename = "$(filename_prefix)-$(timestamp).jld2"
                println("Saving progress to checkpoint file $filename.")
                println()
                flush(stdout)
                save_object(filename, opt)
                last_save_ns += SAVE_INTERVAL_NS
                if !isnothing(last_save_file)
                    rm(last_save_file)
                end
                last_save_file = filename
            end
            if time_ns() - start_time_ns >= 28_800_000_000_000
                break
            end
        end
    end

    println("Final checkpoint file: ", last_save_file)
    println("Exiting after ", time_ns() - start_time_ns, " ns.")
end


main()
