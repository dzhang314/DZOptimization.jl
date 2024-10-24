using Dates: format, now
using DZOptimization.SortingNetworks
using JLD2: save_object


const SAVE_INTERVAL_NS = UInt64(60_000_000_000)
const N = parse(Int, ARGS[2])


if ARGS[1] == "W"
    const CONDITION_CODE = 'W'
    const CONDITION_TYPE = WeaklyNormalizedCondition{2 * N,N}
elseif ARGS[1] == "S"
    const CONDITION_CODE = 'S'
    const CONDITION_TYPE = StronglyNormalizedCondition{2 * N,N}
elseif ARGS[1] == "C"
    const CONDITION_CODE = 'C'
    const CONDITION_TYPE = CompletelyNormalizedCondition{2 * N,N}
else
    error("Invalid condition code $(ARGS[1]) (must be W, S, or C)")
end


function main()

    gen = MultiFloatTestGenerator{2 * N,N,N}()
    cond = CONDITION_TYPE()
    opt = SortingNetworkOptimizer(gen, cond; pareto_radius=1)

    last_save_ns = time_ns()
    last_save_file = nothing
    filename_prefix = "Search$(CONDITION_CODE)$(N)"

    if length(ARGS) >= 3
        @assert startswith(ARGS[3], filename_prefix)
        @assert endswith(ARGS[3], ".jld2")
        opt = load_object(ARGS[3])
        @assert opt isa SortingNetworkOptimizer{N,Float64,
            MultiFloatTestGenerator{2 * N,N,N},CONDITION_TYPE}
        assert_valid(opt)
    end

    while true

        step!(opt; verbose=true)
        println()
        flush(stdout)

        if !isempty(opt.pareto_frontier)
            counts = Dict{Tuple{Int,Int},Int}()
            for (network, _) in opt.passing_networks
                point = (length(network), depth(network))
                if haskey(counts, point)
                    counts[point] += 1
                else
                    counts[point] = 1
                end
            end
            println([point => counts[point]
                     for point in sort!(collect(opt.pareto_frontier))])
            println()
            flush(stdout)
        end

        if time_ns() - last_save_ns >= SAVE_INTERVAL_NS
            disable_sigint() do
                timestamp = format(now(), "yyyy-mm-dd-HH-MM-SS")
                filename = "$(filename_prefix)-$(timestamp).jld2"
                println("Saving progress to file $filename.")
                println()
                flush(stdout)
                save_object(filename, opt)
                last_save_ns += SAVE_INTERVAL_NS
                if !isnothing(last_save_file)
                    rm(last_save_file)
                end
                last_save_file = filename
            end
        end
    end
end


main()
