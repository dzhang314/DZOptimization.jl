module SortingNetworks


using Base.Threads: Atomic, nthreads, @spawn


################################################# SORTING NETWORK DATA STRUCTURE


export SortingNetwork, canonize!


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
    a.comparators == b.comparators
@inline Base.hash(network::SortingNetwork{N}, h::UInt) where {N} =
    hash(network.comparators, h)


# function depth(network::SortingNetwork{N}) where {N}
#     Base.require_one_based_indexing(network.comparators)
#     if isempty(network.comparators)
#         return 0
#     else
#         result = 1
#         for i = 1:length(network.comparators)-1
#             @inbounds (a, b) = network.comparators[i]
#             @inbounds (c, d) = network.comparators[i+1]
#             @assert a < b
#             @assert c < d
#             if (a == c) | (a == d) | (b == c) | (b == d)
#                 result += 1
#             end
#         end
#         return result
#     end
# end


function canonize!(network::SortingNetwork{N}) where {N}
    Base.require_one_based_indexing(network.comparators)
    while true
        changed = false
        for i = 1:length(network.comparators)-1
            @inbounds (a, b) = network.comparators[i]
            @inbounds (c, d) = network.comparators[i+1]
            @assert (a < b) & (c < d)
            if (a != c) & (a != d) & (b != c) & (b != d) & ((a, b) > (c, d))
                @inbounds network.comparators[i] = (c, d)
                @inbounds network.comparators[i+1] = (a, b)
                changed = true
            end
        end
        if !changed
            return network
        end
    end
end


###################################################### SORTING NETWORK EXECUTION


export apply_sort!, apply_two_sum!


@inline _branch_free_minmax(x::T, y::T) where {T} =
    ifelse(x > y, (y, x), (x, y))


@inline function _two_sum(a::T, b::T) where {T}
    s = a + b
    a_eff = s - b
    b_eff = s - a_eff
    a_err = a - a_eff
    b_err = b - b_eff
    e = a_err + b_err
    return (s, e)
end


function apply_sort!(
    x::AbstractVector{T},
    network::SortingNetwork{N},
) where {T,N}
    Base.require_one_based_indexing(x)
    @assert length(x) == N
    for (i, j) in network.comparators
        @inbounds x[i], x[j] = _branch_free_minmax(x[i], x[j])
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
        @inbounds x[i], x[j] = _two_sum(x[i], x[j])
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
        @inbounds x[i], x[j] = _branch_free_minmax(x[i], x[j])
    end
    for k = index+1:length(network.comparators)
        @inbounds i, j = network.comparators[k]
        @inbounds x[i], x[j] = _branch_free_minmax(x[i], x[j])
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
        @inbounds x[i], x[j] = _two_sum(x[i], x[j])
    end
    for k = index+1:length(network.comparators)
        @inbounds i, j = network.comparators[k]
        @inbounds x[i], x[j] = _two_sum(x[i], x[j])
    end
    return x
end


#################################################### SORTING NETWORK COMPILATION


export compile_sort, compile_two_sum


_meta_sort(a::Symbol, b::Symbol) = Expr(:(=),
    Expr(:tuple, a, b), Expr(:call, :_branch_free_minmax, a, b))


_meta_two_sum(a::Symbol, b::Symbol) = Expr(:(=),
    Expr(:tuple, a, b), Expr(:call, :_two_sum, a, b))


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
    test_case::NTuple{N,T},
    cond::AbstractSortingCondition{N},
    network::SortingNetwork{N},
) where {N,T}
    v = collect(test_case)
    apply_sort!(v, network)
    return cond(v)
end


function passes_test(
    test_case::NTuple{N,T},
    cond::AbstractTwoSumCondition{N},
    network::SortingNetwork{N},
) where {N,T}
    v = collect(test_case)
    apply_two_sum!(v, network)
    return cond(v)
end


function passes_test(
    test_case::AbstractVector{T},
    cond::AbstractSortingCondition{N},
    network::SortingNetwork{N},
) where {T,N}
    v = Vector{T}(undef, N)
    copy!(v, test_case)
    apply_sort!(v, network)
    return cond(v)
end


function passes_test(
    test_case::AbstractVector{T},
    cond::AbstractTwoSumCondition{N},
    network::SortingNetwork{N},
) where {T,N}
    v = Vector{T}(undef, N)
    copy!(v, test_case)
    apply_two_sum!(v, network)
    return cond(v)
end


function passes_all_tests(
    test_cases::AbstractSet{NTuple{N,T}},
    cond::AbstractSortingCondition{N},
    network::SortingNetwork{N},
) where {N,T}
    v = Vector{T}(undef, N)
    for test_case in test_cases
        @simd ivdep for i = 1:N
            @inbounds v[i] = test_case[i]
        end
        apply_sort!(v, network)
        if !cond(v)
            return false
        end
    end
    return true
end


function passes_all_tests(
    test_cases::AbstractSet{NTuple{N,T}},
    cond::AbstractTwoSumCondition{N},
    network::SortingNetwork{N},
) where {N,T}
    v = Vector{T}(undef, N)
    for test_case in test_cases
        @simd ivdep for i = 1:N
            @inbounds v[i] = test_case[i]
        end
        apply_two_sum!(v, network)
        if !cond(v)
            return false
        end
    end
    return true
end


function passes_all_tests(
    test_cases::AbstractSet{V},
    cond::AbstractSortingCondition{N},
    network::SortingNetwork{N},
) where {T,V<:AbstractVector{T},N}
    v = Vector{T}(undef, N)
    for test_case in test_cases
        copy!(v, test_case)
        apply_sort!(v, network)
        if !cond(v)
            return false
        end
    end
    return true
end


function passes_all_tests(
    test_cases::AbstractSet{V},
    cond::AbstractTwoSumCondition{N},
    network::SortingNetwork{N},
) where {T,V<:AbstractVector{T},N}
    v = Vector{T}(undef, N)
    for test_case in test_cases
        copy!(v, test_case)
        apply_two_sum!(v, network)
        if !cond(v)
            return false
        end
    end
    return true
end


function _passes_all_tests_without(
    test_cases::AbstractSet{NTuple{N,T}},
    cond::AbstractSortingCondition{N},
    network::SortingNetwork{N},
    index::Int,
) where {N,T}
    v = Vector{T}(undef, N)
    for test_case in test_cases
        @simd ivdep for i = 1:N
            @inbounds v[i] = test_case[i]
        end
        _apply_sort_without!(v, network, index)
        if !cond(v)
            return false
        end
    end
    return true
end


function _passes_all_tests_without(
    test_cases::AbstractSet{NTuple{N,T}},
    cond::AbstractTwoSumCondition{N},
    network::SortingNetwork{N},
    index::Int,
) where {N,T}
    v = Vector{T}(undef, N)
    for test_case in test_cases
        @simd ivdep for i = 1:N
            @inbounds v[i] = test_case[i]
        end
        _apply_two_sum_without!(v, network, index)
        if !cond(v)
            return false
        end
    end
    return true
end


function _passes_all_tests_without(
    test_cases::AbstractSet{V},
    cond::AbstractSortingCondition{N},
    network::SortingNetwork{N},
    index::Int,
) where {T,V<:AbstractVector{T},N}
    v = Vector{T}(undef, N)
    for test_case in test_cases
        copy!(v, test_case)
        _apply_sort_without!(v, network, index)
        if !cond(v)
            return false
        end
    end
    return true
end


function _passes_all_tests_without(
    test_cases::AbstractSet{V},
    cond::AbstractTwoSumCondition{N},
    network::SortingNetwork{N},
    index::Int,
) where {T,V<:AbstractVector{T},N}
    v = Vector{T}(undef, N)
    for test_case in test_cases
        copy!(v, test_case)
        _apply_two_sum_without!(v, network, index)
        if !cond(v)
            return false
        end
    end
    return true
end


##################################################### SORTING NETWORK GENERATION


export generate_sorting_network


@inline function _random_comparator(num_inputs::UInt8)
    i = rand(Base.OneTo(num_inputs))
    j = rand(Base.OneTo(num_inputs - one(UInt8)))
    j += (j >= i)
    return branch_free_minmax(i, j)
end


function generate_sorting_network(
    test_cases::AbstractSet{NTuple{N,T}},
    cond::AbstractCondition{N},
) where {N,T}

    # Generate a random sorting network by adding random comparators
    # until the network satisfies the given condition on every test case.
    network = SortingNetwork{N}(Tuple{UInt8,UInt8}[])
    while !passes_all_tests(test_cases, cond, network)
        push!(network.comparators, _random_comparator(num_inputs))
    end

    # Prune the network by removing unnecessary comparators.
    while !isempty(network.comparators)
        pruned = false
        for index = 1:length(network.comparators)
            if _passes_all_tests_without(test_cases, cond, network, index)
                deleteat!(network.comparators, index)
                pruned = true
                break
            end
        end
        if !pruned
            break
        end
    end

    canonize!(network)
    return network
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
    duration_ns::UInt64,
    num_threads::Int=nthreads(),
) where {N,T}
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
    Base.require_one_based_indexing(x)
    @inbounds for i = 1:M-1
        if !_is_weakly_normalized(x[i], x[i+1])
            return false
        end
    end
    @inbounds first_limb = x[1]
    @inbounds for i = M+1:length(x)
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
    Base.require_one_based_indexing(x)
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


################################################################ RENORMALIZATION


function _renormalize!(v::AbstractVector{T}) where {T}
    Base.require_one_based_indexing(v)
    while true
        changed = false
        for i = 1:length(v)-1
            @inbounds x, y = v[i], v[i+1]
            (s, e) = _two_sum(x, y)
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


#=


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
    n::Integer=3,
)
    Base.require_one_based_indexing(line)
    for i = 1:length(line)-1
        @inbounds a, b = line[i], line[i+1]
        padding = UNICODE_PADDING_CHARACTERS[(a, b)]
        print(io, a)
        for _ = Base.OneTo(n)
            print(io, padding)
        end
    end
    print(io, line[end], '\n')
end


function println_unicode(io::IO, network::SortingNetwork; n::Integer=3)
    line = fill(Char(0x2502), network.num_inputs)
    for (i, j) in network.comparators
        @assert i < j
        if any(line[k] != Char(0x2502) for k = i:j)
            println_padded_unicode(io, line; n)
            fill!(line, Char(0x2502))
        end
        line[i] = Char(0x255E)
        line[j] = Char(0x2561)
        @simd ivdep for k = i+1:j-1
            @inbounds line[k] = Char(0x256A)
        end
    end
    if any(c != Char(0x2502) for c in line)
        println_padded_unicode(io, line; n)
    end
end


println_unicode(network::SortingNetwork; n::Integer=3) =
    println_unicode(stdout, network; n)


=#


end # module SortingNetworks
