module SortingNetworks


using Base.Threads: Atomic, nthreads, @threads, @spawn
using Random: shuffle
import ..assert_valid, ..step!, ..branch_free_minmax, ..two_sum


################################################# SORTING NETWORK DATA STRUCTURE


export SortingNetwork


struct SortingNetwork{N}
    comparators::Vector{Tuple{UInt8,UInt8}}

    function SortingNetwork{N}(
        comparators::Vector{Tuple{UInt8,UInt8}},
    ) where {N}
        @assert !signbit(N)
        @assert N <= typemax(UInt8)
        _zero = zero(UInt8)
        _num_inputs = UInt8(N)
        for (a, b) in comparators
            @assert _zero < a < b <= _num_inputs
        end
        return new{N}(comparators)
    end
end


@inline Base.length(network::SortingNetwork{N}) where {N} =
    length(network.comparators)
@inline Base.:(==)(a::SortingNetwork{N}, b::SortingNetwork{N}) where {N} =
    (a.comparators == b.comparators)
@inline Base.hash(network::SortingNetwork{N}, h::UInt) where {N} =
    hash(network.comparators, h)


####################################################### SORTING NETWORK ANALYSIS


export assert_valid, canonize, sort_fitness, two_sum_fitness


struct Instruction{T}
    opcode::Symbol
    outputs::Vector{T}
    inputs::Vector{T}
end


const AbstractVecOrSet{T} = Union{AbstractVector{T},AbstractSet{T}}


function assert_valid(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVecOrSet{T},
    inputs::AbstractVecOrSet{T},
) where {T}
    computed = Set{T}(inputs)
    for instr in code
        for input in instr.inputs
            @assert input in computed
        end
        for output in instr.outputs
            @assert !(output in computed)
            push!(computed, output)
        end
    end
    @assert issubset(outputs, computed)
    return true
end


function eliminate_dead_code!(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVecOrSet{T},
) where {T}
    if !isempty(code)
        needed = Set{T}(outputs)
        dead_indices = BitSet()
        for (index, instr) in Iterators.reverse(pairs(code))
            if any(output in needed for output in instr.outputs)
                for input in instr.inputs
                    push!(needed, input)
                end
            else
                push!(dead_indices, index)
            end
        end
        deleteat!(code, dead_indices)
    end
    return code
end


function canonize_code(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVecOrSet{T},
    inputs::AbstractVecOrSet{T},
) where {T}
    eliminate_dead_code!(code, outputs)
    if isempty(code)
        return code
    end
    generation = Dict{T,Int}()
    blocks = Vector{Vector{Instruction{T}}}()
    for input in inputs
        generation[input] = 0
    end
    for instr in code
        gen = 0
        for input in instr.inputs
            @assert haskey(generation, input)
            gen = max(gen, generation[input])
        end
        gen += 1
        if gen <= length(blocks)
            push!(blocks[gen], instr)
        else
            @assert gen == length(blocks) + 1
            push!(blocks, [instr])
        end
        for output in instr.outputs
            @assert !haskey(generation, output)
            generation[output] = gen
        end
    end
    for block in blocks
        sort!(block, by=instr -> instr.outputs)
    end
    return reduce(vcat, blocks)
end


function code_depth(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVecOrSet{T},
    inputs::AbstractVecOrSet{T},
) where {T}
    generation = Dict{T,Int}()
    for input in inputs
        generation[input] = 0
    end
    for instr in code
        gen = 0
        for input in instr.inputs
            @assert haskey(generation, input)
            gen = max(gen, generation[input])
        end
        gen += 1
        for output in instr.outputs
            @assert !haskey(generation, output)
            generation[output] = gen
        end
    end
    result = 0
    for output in outputs
        @assert haskey(generation, output)
        result = max(result, generation[output])
    end
    return result
end


function sort_code(
    network::SortingNetwork{N},
    num_outputs::Int,
) where {N}
    generation = [1 for _ = 1:N]
    code = Instruction{Tuple{Int,Int}}[]
    for (x, y) in network.comparators
        @assert x < y
        x = Int(x)
        y = Int(y)
        i = generation[x]
        j = generation[y]
        i_next = i + 1
        j_next = j + 1
        push!(code, Instruction(:minmax,
            [(x, i_next), (y, j_next)], [(x, i), (y, j)]))
        generation[x] = i_next
        generation[y] = j_next
    end
    outputs = [(i, generation[i]) for i = 1:num_outputs]
    inputs = [(i, 1) for i = 1:N]
    return (code, outputs, inputs)
end


function canonize(network::SortingNetwork{N}) where {N}
    code, outputs, inputs = sort_code(network, N)
    return SortingNetwork{N}([
        (UInt8(instr.outputs[1][1]), UInt8(instr.outputs[2][1]))
        for instr in canonize_code(code, outputs, inputs)])
end


function sort_fitness(
    network::SortingNetwork{N},
    num_outputs::Int,
) where {N}
    code, outputs, inputs = sort_code(network, num_outputs)
    eliminate_dead_code!(code, outputs)
    return (length(code), code_depth(code, outputs, inputs))
end


function two_sum_code(
    network::SortingNetwork{N},
    num_outputs::Int,
) where {N}
    generation = [1 for _ = 1:N]
    code = Instruction{Tuple{Int,Int}}[]
    k = N
    for (x, y) in network.comparators
        @assert x < y
        x = Int(x)
        y = Int(y)
        i = generation[x]
        j = generation[y]
        i_next = i + 1
        j_next = j + 1
        k += 1
        push!(code, Instruction(:+, [(x, i_next)], [(x, i), (y, j)]))
        push!(code, Instruction(:-, [(k, 1)], [(x, i_next), (y, j)]))
        push!(code, Instruction(:-, [(k, 2)], [(x, i_next), (k, 1)]))
        push!(code, Instruction(:-, [(k, 3)], [(x, i), (k, 1)]))
        push!(code, Instruction(:-, [(k, 4)], [(y, j), (k, 2)]))
        push!(code, Instruction(:+, [(y, j_next)], [(k, 3), (k, 4)]))
        generation[x] = i_next
        generation[y] = j_next
    end
    outputs = [(i, generation[i]) for i = 1:num_outputs]
    inputs = [(i, 1) for i = 1:N]
    return (code, outputs, inputs)
end


function two_sum_fitness(
    network::SortingNetwork{N},
    num_outputs::Int,
) where {N}
    code, outputs, inputs = two_sum_code(network, num_outputs)
    eliminate_dead_code!(code, outputs)
    return (length(code), code_depth(code, outputs, inputs))
end


###################################################### SORTING NETWORK EXECUTION


export apply_sort!, apply_two_sum!


function apply_sort!(
    x::AbstractVector{T},
    network::SortingNetwork{N},
) where {T,N}
    Base.require_one_based_indexing(x)
    @assert length(x) == N
    for (i, j) in network.comparators
        @inbounds x[i], x[j] = branch_free_minmax(x[i], x[j])
    end
    return x
end


function apply_two_sum!(
    x::AbstractVector{T},
    network::SortingNetwork{N},
) where {T,N}
    Base.require_one_based_indexing(x)
    @assert length(x) == N
    for (i, j) in network.comparators
        @inbounds x[i], x[j] = two_sum(x[i], x[j])
    end
    return x
end


function _apply_sort_without!(
    x::AbstractVector{T},
    network::SortingNetwork{N},
    index::Int,
) where {T,N}
    Base.require_one_based_indexing(x)
    @assert length(x) == N
    @assert 1 <= index <= length(network.comparators)
    for k = 1:index-1
        @inbounds i, j = network.comparators[k]
        @inbounds x[i], x[j] = branch_free_minmax(x[i], x[j])
    end
    for k = index+1:length(network.comparators)
        @inbounds i, j = network.comparators[k]
        @inbounds x[i], x[j] = branch_free_minmax(x[i], x[j])
    end
    return x
end


function _apply_two_sum_without!(
    x::AbstractVector{T},
    network::SortingNetwork{N},
    index::Int,
) where {T,N}
    Base.require_one_based_indexing(x)
    @assert length(x) == N
    @assert 1 <= index <= length(network.comparators)
    for k = 1:index-1
        @inbounds i, j = network.comparators[k]
        @inbounds x[i], x[j] = two_sum(x[i], x[j])
    end
    for k = index+1:length(network.comparators)
        @inbounds i, j = network.comparators[k]
        @inbounds x[i], x[j] = two_sum(x[i], x[j])
    end
    return x
end


#################################################### SORTING NETWORK COMPILATION


export compile_sort, compile_two_sum


_meta_sort(a::Symbol, b::Symbol) = Expr(:(=),
    Expr(:tuple, a, b), Expr(:call, :branch_free_minmax, a, b))


_meta_two_sum(a::Symbol, b::Symbol) = Expr(:(=),
    Expr(:tuple, a, b), Expr(:call, :two_sum, a, b))


function compile_sort(network::SortingNetwork{N}) where {N}
    xs = [Symbol('x', i) for i in Base.OneTo(N)]
    body = Expr[]
    push!(body, Expr(:meta, :inline))
    push!(body, Expr(:(=), Expr(:tuple, xs...), :x))
    for (i, j) in network.comparators
        @assert i < j
        push!(body, _meta_sort(xs[i], xs[j]))
    end
    push!(body, Expr(:return, Expr(:tuple, xs...)))
    return eval(Expr(:->, :x, Expr(:block, body...)))
end


function compile_two_sum(network::SortingNetwork{N}) where {N}
    xs = [Symbol('x', i) for i in Base.OneTo(N)]
    body = Expr[]
    push!(body, Expr(:meta, :inline))
    push!(body, Expr(:(=), Expr(:tuple, xs...), :x))
    for (i, j) in network.comparators
        @assert i < j
        push!(body, _meta_two_sum(xs[i], xs[j]))
    end
    push!(body, Expr(:return, Expr(:tuple, xs...)))
    return eval(Expr(:->, :x, Expr(:block, body...)))
end


######################################################## SORTING NETWORK TESTING


export passes_test, passes_all_tests


abstract type AbstractCondition{N} end


abstract type AbstractSortingCondition{N} <: AbstractCondition{N} end


abstract type AbstractTwoSumCondition{N} <: AbstractCondition{N} end


function passes_test(
    test_case, # intentionally duck-typed
    cond::AbstractSortingCondition{N},
    network::SortingNetwork{N},
) where {N}
    @assert length(test_case) == N
    temp = collect(test_case)
    apply_sort!(temp, network)
    return cond(temp)
end


function passes_test(
    test_case, # intentionally duck-typed
    cond::AbstractTwoSumCondition{N},
    network::SortingNetwork{N},
) where {N}
    @assert length(test_case) == N
    temp = collect(test_case)
    apply_two_sum!(temp, network)
    return cond(temp)
end


function _unsafe_passes_test!(
    temp::Vector{T},
    test_case, # intentionally duck-typed
    cond::AbstractSortingCondition{N},
    network::SortingNetwork{N},
) where {T,N}
    @simd ivdep for i = 1:N
        @inbounds temp[i] = test_case[i]
    end
    apply_sort!(temp, network)
    return cond(temp)
end


function _unsafe_passes_test!(
    temp::Vector{T},
    test_case, # intentionally duck-typed
    cond::AbstractTwoSumCondition{N},
    network::SortingNetwork{N},
) where {T,N}
    @simd ivdep for i = 1:N
        @inbounds temp[i] = test_case[i]
    end
    apply_two_sum!(temp, network)
    return cond(temp)
end


function _unsafe_passes_test_without!(
    temp::Vector{T},
    test_case, # intentionally duck-typed
    cond::AbstractSortingCondition{N},
    network::SortingNetwork{N},
    index::Int,
) where {T,N}
    @simd ivdep for i = 1:N
        @inbounds temp[i] = test_case[i]
    end
    _apply_sort_without!(temp, network, index)
    return cond(temp)
end


function _unsafe_passes_test_without!(
    temp::Vector{T},
    test_case, # intentionally duck-typed
    cond::AbstractTwoSumCondition{N},
    network::SortingNetwork{N},
    index::Int,
) where {T,N}
    @simd ivdep for i = 1:N
        @inbounds temp[i] = test_case[i]
    end
    _apply_two_sum_without!(temp, network, index)
    return cond(temp)
end


function passes_all_tests(
    test_cases::AbstractVecOrSet{NTuple{N,T}},
    cond::AbstractCondition{N},
    network::SortingNetwork{N},
) where {N,T}
    temp = Vector{T}(undef, N)
    for test_case in test_cases
        if !_unsafe_passes_test!(temp, test_case, cond, network)
            return false
        end
    end
    return true
end


function _passes_all_tests_without(
    test_cases::AbstractVecOrSet{NTuple{N,T}},
    cond::AbstractCondition{N},
    network::SortingNetwork{N},
    index::Int,
) where {N,T}
    temp = Vector{T}(undef, N)
    for test_case in test_cases
        if !_unsafe_passes_test_without!(temp, test_case, cond, network, index)
            return false
        end
    end
    return true
end


##################################################### SORTING NETWORK GENERATION


export random_insert!, random_replace!, random_swap!,
    removable_comparators, prune!, generate_sorting_network, generate_mutation


@inline function _random_comparator(num_inputs::UInt8)
    i = rand(Base.OneTo(num_inputs))
    j = rand(Base.OneTo(num_inputs - one(UInt8)))
    j += (j >= i)
    return branch_free_minmax(i, j)
end


function random_insert!(network::SortingNetwork{N}) where {N}
    index = rand(Base.OneTo(length(network.comparators) + 1))
    insert!(network.comparators, index, _random_comparator(UInt8(N)))
    return (network, index)
end


function random_replace!(network::SortingNetwork{N}) where {N}
    index = rand(Base.OneTo(length(network.comparators)))
    network.comparators[index] = _random_comparator(UInt8(N))
    return (network, index)
end


function random_swap!(network::SortingNetwork{N}) where {N}
    i = rand(Base.OneTo(length(network.comparators)))
    j = rand(Base.OneTo(length(network.comparators) - 1))
    j += (j >= i)
    @inbounds network.comparators[i], network.comparators[j] =
        network.comparators[j], network.comparators[i]
    return (network, (i, j))
end


removable_comparators(
    test_cases::AbstractVecOrSet{NTuple{N,T}},
    cond::AbstractCondition{N},
    network::SortingNetwork{N},
) where {N,T} = BitSet(
    index for index in eachindex(network.comparators)
    if _passes_all_tests_without(test_cases, cond, network, index))


function prune!(
    network::SortingNetwork{N},
    test_cases::AbstractVecOrSet{NTuple{N,T}},
    cond::AbstractCondition{N},
) where {N,T}
    while true
        removable_indices = removable_comparators(test_cases, cond, network)
        if isempty(removable_indices)
            return network
        end
        deleteat!(network.comparators, rand(removable_indices))
    end
end


function generate_sorting_network(
    test_cases::AbstractVecOrSet{NTuple{N,T}},
    cond::AbstractCondition{N},
) where {N,T}

    # Generate a random sorting network by adding random comparators
    # until the network satisfies the given condition on every test case.
    _num_inputs = UInt8(N)
    network = SortingNetwork{N}(Tuple{UInt8,UInt8}[])
    while !passes_all_tests(test_cases, cond, network)
        push!(network.comparators, _random_comparator(_num_inputs))
    end

    # Prune the network by removing unnecessary comparators.
    return prune!(network, test_cases, cond)
end


function generate_mutation(
    network::SortingNetwork{N},
    test_cases::AbstractVecOrSet{NTuple{N,T}},
    cond::AbstractCondition{N};
    insertion_radius::Int,
    replacement_radius::Int,
    swap_radius::Int,
) where {N,T}
    @assert !signbit(insertion_radius)
    @assert !signbit(replacement_radius)
    @assert !signbit(swap_radius)
    while true
        new_network, _ = random_insert!(deepcopy(network))
        for _ = 1:rand(0:insertion_radius)
            random_insert!(new_network)
        end
        for _ = 1:rand(0:replacement_radius)
            random_replace!(new_network)
        end
        for _ = 1:rand(0:swap_radius)
            random_swap!(new_network)
        end
        prune!(new_network, test_cases, cond)
        if passes_all_tests(test_cases, cond, new_network)
            return new_network
        end
    end
end


############################################################ TEST CASE SELECTION


# export necessary_test_cases


# function necessary_test_cases(
#     test_cases::Set{Vector{T}},
#     cond::AbstractCondition{N},
#     networks::Set{SortingNetwork},
# ) where {T,N}

#     # Compute the set of networks that each test case invalidates.
#     invalidation_sets = Dict{Vector{T},Set{SortingNetwork}}()
#     for test_case in test_cases
#         invalidated = Set{SortingNetwork}()
#         for network in networks
#             if !passes_test(test_case, cond, network)
#                 push!(invalidated, network)
#             end
#         end
#         invalidation_sets[test_case] = invalidated
#     end

#     # Greedily compute a set covering of the invalidated networks.
#     result = Set{Vector{Float64}}()
#     while !all(isempty, values(invalidation_sets))
#         _, best_test_case = findmax(length, invalidation_sets)
#         push!(result, best_test_case)
#         for other_test_case in test_cases
#             if other_test_case != best_test_case
#                 setdiff!(invalidation_sets[other_test_case],
#                     invalidation_sets[best_test_case])
#             end
#         end
#         empty!(invalidation_sets[best_test_case])
#     end

#     return result
# end


########################################################## COUNTEREXAMPLE SEARCH


export find_counterexample


abstract type AbstractTestGenerator{N,T} end


abstract type AbstractSortingTestGenerator{N,T} <:
              AbstractTestGenerator{N,T} end


abstract type AbstractTwoSumTestGenerator{N,T} <:
              AbstractTestGenerator{N,T} end


function _find_counterexample_worker_thread(
    gen::AbstractTestGenerator{N,T},
    cond::AbstractCondition{N},
    compiled_network::Function,
    terminate::Atomic{Bool},
) where {N,T}
    num_tries = zero(UInt64)
    while !terminate[]
        num_tries += 1
        test_case = gen()
        if !cond(compiled_network(test_case))
            terminate[] = true
            return (test_case, num_tries)
        end
    end
    return (nothing, num_tries)
end


function _find_counterexample_main_thread(
    gen::AbstractTestGenerator{N,T},
    cond::AbstractCondition{N},
    compiled_network::Function,
    terminate::Atomic{Bool},
    duration_ns::UInt64,
) where {N,T}
    start = time_ns()
    num_tries = zero(UInt64)
    while !terminate[]
        num_tries += 1
        test_case = gen()
        if !cond(compiled_network(test_case))
            terminate[] = true
            return (test_case, num_tries)
        end
        if time_ns() - start >= duration_ns
            terminate[] = true
            return (nothing, num_tries)
        end
    end
    return (nothing, num_tries)
end


_compile_helper(
    ::AbstractSortingTestGenerator{N,T},
    ::AbstractSortingCondition{N},
    network::SortingNetwork{N},
) where {N,T} = compile_sort(network)


_compile_helper(
    ::AbstractTwoSumTestGenerator{N,T},
    ::AbstractTwoSumCondition{N},
    network::SortingNetwork{N},
) where {N,T} = compile_two_sum(network)


function find_counterexample(
    gen::AbstractTestGenerator{N,T},
    cond::AbstractCondition{N},
    network::SortingNetwork{N},
    duration_ns::UInt64;
    num_threads::Int=nthreads(),
) where {N,T}
    if iszero(duration_ns)
        return (nothing, zero(UInt64))
    end
    @assert num_threads >= 1
    terminate = Atomic{Bool}(false)
    compiled_network = _compile_helper(gen, cond, network)
    tasks = Task[
        @spawn invokelatest(_find_counterexample_worker_thread,
            $(deepcopy(gen)), $(deepcopy(cond)), $compiled_network, $terminate
        )::Tuple{Union{Nothing,NTuple{N,T}},UInt64}
        for _ = 1:num_threads-1]
    results = Tuple{Union{Nothing,NTuple{N,T}},UInt64}[]
    push!(results, invokelatest(_find_counterexample_main_thread,
        gen, cond, compiled_network, terminate, duration_ns
    )::Tuple{Union{Nothing,NTuple{N,T}},UInt64})
    for task in tasks
        push!(results, fetch(task)::Tuple{Union{Nothing,NTuple{N,T}},UInt64})
    end
    num_tries = zero(UInt64)
    for i = 1:num_threads
        num_tries += results[i][2]
    end
    for i = 1:num_threads
        if !isnothing(results[i][1])
            return (results[i][1], num_tries)
        end
    end
    return (nothing, num_tries)
end


####################################################### OPTIMIZER DATA STRUCTURE


export SortingNetworkOptimizer, fitness, assert_valid, step!


struct SortingNetworkOptimizer{
    N,T,G<:AbstractTestGenerator{N,T},C<:AbstractCondition{N}}
    test_cases::Vector{NTuple{N,T}}
    gen::G
    cond::C
    num_outputs::Int
    passing_networks::Dict{SortingNetwork{N},UInt64}
    failing_networks::Dict{SortingNetwork{N},Int}
    failure_sets::Vector{Pair{SortingNetwork{N},BitSet}}
    pareto_frontier::Set{Tuple{Int,Int}}
    pareto_radius::Int
end


function SortingNetworkOptimizer(
    gen::G,
    cond::C,
    num_outputs::Int;
    pareto_radius::Integer=0,
) where {N,T,G<:AbstractTestGenerator{N,T},C<:AbstractCondition{N}}
    @assert !signbit(num_outputs)
    @assert !signbit(pareto_radius)
    return SortingNetworkOptimizer{N,T,G,C}(
        NTuple{N,T}[], gen, cond, num_outputs,
        Dict{SortingNetwork{N},UInt64}(),
        Dict{SortingNetwork{N},Int}(),
        Pair{SortingNetwork{N},BitSet}[],
        Set{Tuple{Int,Int}}(), Int(pareto_radius))
end


@inline _strictly_dominates(
    (len_a, dep_a)::Tuple{Int,Int},
    (len_b, dep_b)::Tuple{Int,Int},
) = (((len_a < len_b) & (dep_a <= dep_b)) |
     ((len_a <= len_b) & (dep_a < dep_b)))


@inline _lies_on_frontier(
    point::Tuple{Int,Int},
    frontier::Set{Tuple{Int,Int}},
) = !any(_strictly_dominates(frontier_point, point)
         for frontier_point in frontier)


@inline _fitness_helper(
    ::AbstractSortingTestGenerator{N,T},
    ::AbstractSortingCondition{N},
    network::SortingNetwork{N},
    num_outputs::Int,
) where {N,T} = sort_fitness(network, num_outputs)


@inline _fitness_helper(
    ::AbstractTwoSumTestGenerator{N,T},
    ::AbstractTwoSumCondition{N},
    network::SortingNetwork{N},
    num_outputs::Int,
) where {N,T} = two_sum_fitness(network, num_outputs)


@inline fitness(
    opt::SortingNetworkOptimizer{N,T,G,C},
    network::SortingNetwork{N},
) where {N,T,G<:AbstractTestGenerator{N,T},C<:AbstractCondition{N}} =
    _fitness_helper(opt.gen, opt.cond, network, opt.num_outputs)


function assert_valid(
    opt::SortingNetworkOptimizer{N,T,G,C},
) where {N,T,G<:AbstractTestGenerator{N,T},C<:AbstractCondition{N}}

    r = opt.pareto_radius
    all_points = Set{Tuple{Int,Int}}()
    for (network, _) in opt.passing_networks
        @assert network == canonize(network)
        @assert passes_all_tests(opt.test_cases, opt.cond, network)
        point = fitness(opt, network)
        push!(all_points, point)
        @assert !any(_strictly_dominates(point, frontier_point)
                     for frontier_point in opt.pareto_frontier)
        @assert _lies_on_frontier(point .- r, opt.pareto_frontier)
    end

    for point in opt.pareto_frontier
        @assert point in all_points
    end

    for (network, index) in opt.failing_networks
        @assert network == canonize(network)
        _network, failure_set = opt.failure_sets[index]
        @assert network === _network
        @assert !isempty(failure_set)
        for (test_index, test_case) in enumerate(opt.test_cases)
            if test_index in failure_set
                @assert !passes_test(test_case, opt.cond, network)
            else
                @assert passes_test(test_case, opt.cond, network)
            end
        end
    end

    for p in opt.pareto_frontier
        for q in opt.pareto_frontier
            @assert (p == q) || !_strictly_dominates(p, q)
        end
    end

    return true
end


function _generate(
    opt::SortingNetworkOptimizer{N,T,G,C},
) where {N,T,G<:AbstractTestGenerator{N,T},C<:AbstractCondition{N}}
    r = opt.pareto_radius
    frontier_networks = Set(
        network for (network, _) in opt.passing_networks
        if _lies_on_frontier(fitness(opt, network) .- r, opt.pareto_frontier))
    terminate = Atomic{Bool}(false)
    result = Ref{Tuple{SortingNetwork{N},Tuple{Int,Int}}}()
    @threads for _ = 1:nthreads()
        while !terminate[]
            network = generate_sorting_network(opt.test_cases, opt.cond)
            point = fitness(opt, network)
            if _lies_on_frontier(point .- r, opt.pareto_frontier)
                terminate[] = true
                result[] = (canonize(network), point)
            end
            if !isempty(frontier_networks)
                network = generate_mutation(
                    rand(frontier_networks), opt.test_cases, opt.cond;
                    insertion_radius=3, replacement_radius=3, swap_radius=3)
                point = fitness(opt, network)
                if _lies_on_frontier(point .- r, opt.pareto_frontier)
                    terminate[] = true
                    result[] = (canonize(network), point)
                end
            end
        end
    end
    return result[]
end


function _rebuild_frontier!(
    opt::SortingNetworkOptimizer{N,T,G,C},
) where {N,T,G<:AbstractTestGenerator{N,T},C<:AbstractCondition{N}}
    all_points = Set(fitness(opt, network)
                     for (network, _) in opt.passing_networks)
    frontier_points = Set(point for point in all_points
                          if _lies_on_frontier(point, all_points))
    r = opt.pareto_radius
    frontier_networks = Set(
        network for (network, _) in opt.passing_networks
        if _lies_on_frontier(fitness(opt, network) .- r, frontier_points))
    obsolete_networks = setdiff(keys(opt.passing_networks), frontier_networks)
    for network in obsolete_networks
        delete!(opt.passing_networks, network)
    end
    empty!(opt.pareto_frontier)
    union!(opt.pareto_frontier, frontier_points)
    return opt
end


function _add_test_case!(
    opt::SortingNetworkOptimizer{N,T,G,C},
    test_case::NTuple{N,T},
) where {N,T,G<:AbstractTestGenerator{N,T},C<:AbstractCondition{N}}
    index = lastindex(opt.test_cases) + 1
    @threads for (network, failure_set) in opt.failure_sets
        if !passes_test(test_case, opt.cond, network)
            push!(failure_set, index)
        end
    end
    invalidated_networks = [network for (network, _) in opt.passing_networks
                            if !passes_test(test_case, opt.cond, network)]
    if !isempty(invalidated_networks)
        for network in invalidated_networks
            @assert passes_all_tests(opt.test_cases, opt.cond, network)
            delete!(opt.passing_networks, network)
            failure_set = BitSet()
            push!(failure_set, index)
            push!(opt.failure_sets, network => failure_set)
            opt.failing_networks[network] = lastindex(opt.failure_sets)
        end
        _rebuild_frontier!(opt)
    end
    push!(opt.test_cases, test_case)
    return invalidated_networks
end


function _vprintln(verbose::Bool, args...)
    if verbose
        println(stdout, args...)
        flush(stdout)
    end
end


function (opt::SortingNetworkOptimizer{N,T,G,C})(
    network::SortingNetwork{N};
    duration_ns::UInt64=UInt64(1_000_000_000),
    verbose::Bool=false,
) where {N,T,G<:AbstractTestGenerator{N,T},C<:AbstractCondition{N}}

    @assert canonize(network) == network
    @assert !haskey(opt.failing_networks, network)

    start = time_ns()
    counterexample, num_tests = find_counterexample(
        opt.gen, opt.cond, network, duration_ns)
    elapsed = (time_ns() - start) / 1.0e9
    _vprintln(verbose, "Performed $num_tests random tests in $elapsed",
        " seconds ($(num_tests / elapsed) tests per second).")

    if isnothing(counterexample)
        if haskey(opt.passing_networks, network)
            _vprintln(verbose,
                "Network passed all tests but was previously discovered.")
            opt.passing_networks[network] += num_tests
        else
            _vprintln(verbose, "Network passed all tests and is novel.")
            opt.passing_networks[network] = num_tests
            _rebuild_frontier!(opt)
        end
    else
        _vprintln(verbose, "Found counterexample.")
        push!(opt.failure_sets, network => BitSet())
        opt.failing_networks[network] = lastindex(opt.failure_sets)
        invalidated_networks = _add_test_case!(opt, counterexample)
        if !isempty(invalidated_networks)
            num_invalid = length(invalidated_networks)
            _vprintln(verbose,
                "Invalidated $num_invalid previously-passing networks.")
        end
    end

    return opt
end


function _variance(::Type{T}, data::Dict{K,V}) where {T,K,V}
    sum_x = zero(T)
    sum_x2 = zero(T)
    for (_, count) in data
        x = T(count)
        sum_x += x
        sum_x2 += abs2(x)
    end
    n = T(length(data))
    return sum_x2 / n - abs2(sum_x / n)
end


function step!(
    opt::SortingNetworkOptimizer{N,T,G,C};
    duration_ns::UInt64=UInt64(1_000_000_000),
    verbose::Bool=false,
) where {N,T,G<:AbstractTestGenerator{N,T},C<:AbstractCondition{N}}
    start = time_ns()
    network, new_point = _generate(opt)
    elapsed = (time_ns() - start) / 1.0e9
    _vprintln(verbose, "Generated $new_point network in $elapsed seconds.")
    opt(network; duration_ns, verbose)
    return opt
end


########################################################## OPTIMIZER COMBINATION


export combine


_construct_invalidation_set(test_case, cond, networks) = BitSet(
    i for i in eachindex(networks)
    if !passes_test(test_case, cond, networks[i]))

_construct_failure_set(test_cases, cond, network) = BitSet(
    i for i in eachindex(test_cases)
    if !passes_test(test_cases[i], cond, network))


function combine(
    optimizers::Vector{SortingNetworkOptimizer{N,T,G,C}},
) where {N,T,G<:AbstractTestGenerator{N,T},C<:AbstractCondition{N}}

    @assert allequal(opt.gen for opt in optimizers)
    gen = first(optimizers).gen
    @assert allequal(opt.cond for opt in optimizers)
    cond = first(optimizers).cond
    @assert allequal(opt.num_outputs for opt in optimizers)
    num_outputs = first(optimizers).num_outputs
    @assert allequal(opt.pareto_radius for opt in optimizers)
    pareto_radius = first(optimizers).pareto_radius
    result = SortingNetworkOptimizer(gen, cond, num_outputs; pareto_radius)

    all_pass_counts = mergewith(+,
        [opt.passing_networks for opt in optimizers]...)
    all_networks = collect(union(
        reduce(union, keys(opt.passing_networks) for opt in optimizers),
        reduce(union, keys(opt.failing_networks) for opt in optimizers)))
    all_test_cases = unique!(sort!(reduce(vcat,
        opt.test_cases for opt in optimizers)))

    sets = [_construct_invalidation_set(test_case, cond, all_networks)
            for test_case in all_test_cases]

    maximal_set_indices = Int[]
    for i in sortperm(sets; by=length, rev=true)
        if !any(issubset(sets[i], sets[j]) for j in maximal_set_indices)
            push!(maximal_set_indices, i)
        end
    end

    for i in eachindex(sets)
        @assert any(issubset(sets[i], sets[j]) for j in maximal_set_indices)
    end

    for i in maximal_set_indices
        for j in maximal_set_indices
            @assert (i == j) || !issubset(sets[i], sets[j])
        end
    end

    for i in maximal_set_indices
        push!(result.test_cases, all_test_cases[i])
    end

    failing_indices = collect(reduce(union, sets))
    for i in failing_indices
        network = all_networks[i]
        set = _construct_failure_set(result.test_cases, cond, network)
        @assert !isempty(set)
        push!(result.failure_sets, network => set)
        result.failing_networks[network] = lastindex(result.failure_sets)
    end

    passing_indices = setdiff(eachindex(all_networks), failing_indices)
    for i in passing_indices
        network = all_networks[i]
        result.passing_networks[network] = all_pass_counts[network]
    end
    _rebuild_frontier!(result)

    return result
end


######################################################### TEST CONDITION: SORTED


export SortedCondition


struct SortedCondition{N} <: AbstractSortingCondition{N} end


@inline (cond::SortedCondition{N})(x::AbstractVector) where {N} = issorted(x)


############################################## TEST CONDITION: WEAKLY NORMALIZED


export WeaklyNormalizedCondition


struct WeaklyNormalizedCondition{N,M} <: AbstractTwoSumCondition{N}
    function WeaklyNormalizedCondition{N,M}() where {N,M}
        @assert N >= M >= 1
        return new{N,M}()
    end
end


@inline _unsafe_exponent(x::T) where {T<:Base.IEEEFloat} = Int(
    (reinterpret(Unsigned, x) & ~Base.sign_mask(T)) >> Base.significand_bits(T)
) - Base.exponent_bias(T)


@inline function _is_weakly_normalized(a::T, b::T) where {T}
    if iszero(b)
        return true
    elseif iszero(a)
        return false
    else
        return _unsafe_exponent(a) >= _unsafe_exponent(b) + precision(T)
    end
end


@inline function _is_weakly_normalized(a::T, b::T, n::Int) where {T}
    if iszero(b)
        return true
    elseif iszero(a)
        return false
    else
        return _unsafe_exponent(a) >= _unsafe_exponent(b) + n * precision(T)
    end
end


function (cond::WeaklyNormalizedCondition{N,M})(
    x::NTuple{N,T},
) where {N,M,T}
    @inbounds for i = 1:M-1
        if !_is_weakly_normalized(x[i], x[i+1])
            return false
        end
    end
    @inbounds first_limb = x[1]
    @inbounds for i = M+1:N
        if !_is_weakly_normalized(first_limb, x[i], M)
            return false
        end
    end
    return true
end


function (cond::WeaklyNormalizedCondition{N,M})(
    x::AbstractVector{T},
) where {N,M,T}
    Base.require_one_based_indexing(x)
    @assert length(x) == N
    @inbounds for i = 1:M-1
        if !_is_weakly_normalized(x[i], x[i+1])
            return false
        end
    end
    @inbounds first_limb = x[1]
    @inbounds for i = M+1:N
        if !_is_weakly_normalized(first_limb, x[i], M)
            return false
        end
    end
    return true
end


######################################## TEST CONDITION: INCOMPLETELY NORMALIZED


export IncompletelyNormalizedCondition


struct IncompletelyNormalizedCondition{N,M,K} <: AbstractTwoSumCondition{N}
    function IncompletelyNormalizedCondition{N,M,K}() where {N,M,K}
        @assert N >= M >= 1
        @assert K >= 0
        return new{N,M,K}()
    end
end


@inline function _is_incompletely_normalized(a::T, b::T, k::Int) where {T}
    if iszero(b)
        return true
    elseif iszero(a)
        return false
    else
        return _unsafe_exponent(a) >= _unsafe_exponent(b) + (precision(T) - k)
    end
end


@inline function _is_incompletely_normalized(
    a::T, b::T, n::Int, k::Int,
) where {T}
    if iszero(b)
        return true
    elseif iszero(a)
        return false
    else
        return (_unsafe_exponent(a) >=
                _unsafe_exponent(b) + n * (precision(T) - k))
    end
end


function (cond::IncompletelyNormalizedCondition{N,M,K})(
    x::NTuple{N,T},
) where {N,M,K,T}
    @inbounds for i = 1:M-1
        if !_is_incompletely_normalized(x[i], x[i+1], K)
            return false
        end
    end
    @inbounds first_limb = x[1]
    @inbounds for i = M+1:N
        if !_is_incompletely_normalized(first_limb, x[i], M, K)
            return false
        end
    end
    return true
end


function (cond::IncompletelyNormalizedCondition{N,M,K})(
    x::AbstractVector{T},
) where {N,M,K,T}
    Base.require_one_based_indexing(x)
    @assert length(x) == N
    @inbounds for i = 1:M-1
        if !_is_incompletely_normalized(x[i], x[i+1], K)
            return false
        end
    end
    @inbounds first_limb = x[1]
    @inbounds for i = M+1:N
        if !_is_incompletely_normalized(first_limb, x[i], M, K)
            return false
        end
    end
    return true
end


############################################ TEST CONDITION: STRONGLY NORMALIZED


export StronglyNormalizedCondition


struct StronglyNormalizedCondition{N,M} <: AbstractTwoSumCondition{N}
    function StronglyNormalizedCondition{N,M}() where {N,M}
        @assert N >= M >= 1
        return new{N,M}()
    end
end


@inline _is_strongly_normalized(a::T, b::T) where {T} =
    (a, b) === two_sum(a, b)


function (cond::StronglyNormalizedCondition{N,M})(
    x::NTuple{N,T},
) where {N,M,T}
    @inbounds for i = 1:M-1
        if !_is_strongly_normalized(x[i], x[i+1])
            return false
        end
    end
    @inbounds final_limb = x[M]
    @inbounds for i = M+1:N
        if !_is_strongly_normalized(final_limb, x[i])
            return false
        end
    end
    return true
end


function (cond::StronglyNormalizedCondition{N,M})(
    x::AbstractVector{T},
) where {N,M,T}
    Base.require_one_based_indexing(x)
    @assert length(x) == N
    @inbounds for i = 1:M-1
        if !_is_strongly_normalized(x[i], x[i+1])
            return false
        end
    end
    @inbounds final_limb = x[M]
    @inbounds for i = M+1:N
        if !_is_strongly_normalized(final_limb, x[i])
            return false
        end
    end
    return true
end


############################################ TEST CONDITION: STRONGLY NORMALIZED


export CompletelyNormalizedCondition


struct CompletelyNormalizedCondition{N,M} <: AbstractTwoSumCondition{N}
    function CompletelyNormalizedCondition{N,M}() where {N,M}
        @assert N >= M >= 1
        return new{N,M}()
    end
end


function (cond::CompletelyNormalizedCondition{N,M})(
    x::NTuple{N,T},
) where {N,M,T}
    r = _top_down_renormalize(x)
    for i = 1:M
        if r[i] !== x[i]
            return false
        end
    end
    return true
end


function (cond::CompletelyNormalizedCondition{N,M})(
    x::AbstractVector{T},
) where {N,M,T}
    r = copy(x)
    _renormalize!(r)
    for i = 1:M
        if r[i] !== x[i]
            return false
        end
    end
    return true
end


################################################################ RENORMALIZATION


function _renormalize!(v::AbstractVector{T}) where {T}
    Base.require_one_based_indexing(v)
    while true
        changed = false
        for i = 1:length(v)-1
            @inbounds x, y = v[i], v[i+1]
            (s, e) = two_sum(x, y)
            changed |= (s, e) !== (x, y)
            @inbounds v[i], v[i+1] = s, e
        end
        if !changed
            return v
        end
    end
end


@generated function _top_down_renorm_pass(x::NTuple{N,T}) where {N,T}
    xs = [Symbol('x', i) for i in Base.OneTo(N)]
    body = Expr[]
    push!(body, Expr(:meta, :inline))
    push!(body, Expr(:(=), Expr(:tuple, xs...), :x))
    _one = one(N)
    for i in Base.OneTo(N - _one)
        push!(body, _meta_two_sum(xs[i], xs[i+_one]))
    end
    push!(body, Expr(:return, Expr(:tuple, xs...)))
    return Expr(:block, body...)
end


@generated function _bottom_up_renorm_pass(x::NTuple{N,T}) where {N,T}
    xs = [Symbol('x', i) for i in Base.OneTo(N)]
    body = Expr[]
    push!(body, Expr(:meta, :inline))
    push!(body, Expr(:(=), Expr(:tuple, xs...), :x))
    _one = one(N)
    for i = (N-_one):-_one:_one
        push!(body, _meta_two_sum(xs[i], xs[i+_one]))
    end
    push!(body, Expr(:return, Expr(:tuple, xs...)))
    return Expr(:block, body...)
end


function _top_down_renormalize(x::NTuple{N,T}) where {N,T}
    while true
        x_next = _top_down_renorm_pass(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


function _bottom_up_renormalize(x::NTuple{N,T}) where {N,T}
    while true
        x_next = _bottom_up_renorm_pass(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


################################################## RANDOM FLOATING-POINT NUMBERS


@inline _deflate_range_52(i::UInt16) =
    ((i << 5) + (i << 4) + (i << 2) + 0x0038) >> 6


@inline function _stretch_low_bits(x::UInt64, i::UInt16)
    _one = one(UInt64)
    bit = (_one << i)
    mask = bit - _one
    value = bit - (x & _one)
    return (x & ~mask) | (value & mask)
end


@inline function _stretch_high_bits(x::UInt64, j::UInt16)
    _one = one(UInt64)
    _high_bit = _one << 51
    _past_bit = _one << 52
    _full_mask = _past_bit - _one
    mask = (_full_mask >> j) << j
    value = _past_bit - ((x & _high_bit) >> 51)
    return (x & ~mask) | (value & mask)
end


function _generate_random_float()
    # Generate random data for sign bit and exponent.
    r = rand(UInt16)
    exponent = r & 0x03FF
    if iszero(exponent)
        return zero(Float64)
    end
    exponent += 0x01FF
    # Generate random data for mantissa.
    m = rand(UInt64)
    i, j = minmax(
        _deflate_range_52(UInt16((m & 0xFC00000000000000) >> 58)),
        _deflate_range_52(UInt16((m & 0x03F0000000000000) >> 52)))
    m &= 0x000FFFFFFFFFFFFF
    m = _stretch_low_bits(m, i)
    m = _stretch_high_bits(m, j)
    return reinterpret(Float64, ((UInt64(r) << 48) & 0x8000000000000000) |
                                (UInt64(exponent) << 52) | m)
end


############################################### TEST CASE GENERATOR: MULTIFLOATS


export MultiFloatTestGenerator


function _riffle!(
    v::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    Base.require_one_based_indexing(v, x, y)
    len_x = length(x)
    len_y = length(y)
    @assert length(v) == len_x + len_y

    # Riffle common elements.
    @simd ivdep for i = 1:min(len_x, len_y)
        _2i = i + i
        @inbounds v[_2i-1] = x[i]
        @inbounds v[_2i] = y[i]
    end

    # Append remaining elements.
    if len_x > len_y
        copyto!(v, len_y + len_y + 1, x, len_y + 1, len_x - len_y)
    elseif len_x < len_y
        copyto!(v, len_x + len_x + 1, y, len_x + 1, len_y - len_x)
    end

    return v
end


@generated function _riffle(x::NTuple{M,T}, y::NTuple{N,T}) where {M,N,T}
    xs = [Symbol('x', i) for i in Base.OneTo(M)]
    ys = [Symbol('y', i) for i in Base.OneTo(N)]
    vs = _riffle!(Vector{Symbol}(undef, M + N), xs, ys)
    return Expr(:block, Expr(:meta, :inline),
        Expr(:(=), Expr(:tuple, xs...), :x),
        Expr(:(=), Expr(:tuple, ys...), :y),
        Expr(:return, Expr(:tuple, vs...)))
end


struct MultiFloatTestGenerator{N,X,Y} <: AbstractTwoSumTestGenerator{N,Float64}
    function MultiFloatTestGenerator{N,X,Y}() where {N,X,Y}
        @assert N == X + Y
        @assert !signbit(X)
        @assert !signbit(Y)
        return new{N,X,Y}()
    end
end


(gen::MultiFloatTestGenerator{N,X,Y})() where {N,X,Y} = _riffle(
    _top_down_renormalize(ntuple(_ -> _generate_random_float(), Val{X}())),
    _top_down_renormalize(ntuple(_ -> _generate_random_float(), Val{Y}())))


################################################## SORTING NETWORK VISUALIZATION


export println_unicode


const UNICODE_PADDING_CHARACTERS = Dict([
    (Char(0x2502), Char(0x2502)) => Char(0x00A0),
    (Char(0x2502), Char(0x255E)) => Char(0x00A0),
    (Char(0x2561), Char(0x2502)) => Char(0x00A0),
    (Char(0x2561), Char(0x255E)) => Char(0x00A0),
    (Char(0x255E), Char(0x2561)) => Char(0x2550),
    (Char(0x255E), Char(0x256A)) => Char(0x2550),
    (Char(0x256A), Char(0x2561)) => Char(0x2550),
    (Char(0x256A), Char(0x256A)) => Char(0x2550),
])


function println_padded_unicode(
    io::IO,
    line::AbstractVector{Char};
    spacing::Integer=3,
)
    Base.require_one_based_indexing(line)
    for i = 1:length(line)-1
        @inbounds a, b = line[i], line[i+1]
        padding = UNICODE_PADDING_CHARACTERS[(a, b)]
        print(io, a)
        for _ = Base.OneTo(spacing)
            print(io, padding)
        end
    end
    print(io, line[end], '\n')
    flush(io)
end


function println_unicode(
    io::IO,
    network::SortingNetwork{N};
    spacing::Integer=3,
) where {N}
    line = fill(Char(0x2502), N)
    for (i, j) in network.comparators
        @assert i < j
        if any(line[k] != Char(0x2502) for k = i:j)
            println_padded_unicode(io, line; spacing)
            fill!(line, Char(0x2502))
        end
        line[i] = Char(0x255E)
        line[j] = Char(0x2561)
        @simd ivdep for k = i+1:j-1
            @inbounds line[k] = Char(0x256A)
        end
    end
    if any(c != Char(0x2502) for c in line)
        println_padded_unicode(io, line; spacing)
    end
end


println_unicode(network::SortingNetwork; spacing::Integer=3) =
    println_unicode(stdout, network; spacing)


end # module SortingNetworks
