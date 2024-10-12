module SortingNetworks


################################################################################


export SortingNetwork, apply_sort!, apply_two_sum!


struct SortingNetwork
    num_inputs::Int
    comparators::Vector{Tuple{UInt8,UInt8}}

    function SortingNetwork(
        num_inputs::Integer,
        comparators::Vector{Tuple{UInt8,UInt8}},
    )
        @assert !signbit(num_inputs)
        @assert num_inputs <= typemax(UInt8)
        _zero = zero(UInt8)
        _num_inputs = UInt8(num_inputs)
        for (a, b) in comparators
            @assert _zero < a <= _num_inputs
            @assert _zero < b <= _num_inputs
        end
        return new(Int(num_inputs), comparators)
    end
end


@inline Base.length(network::SortingNetwork) = length(network.comparators)
@inline Base.:(==)(a::SortingNetwork, b::SortingNetwork) =
    (a.num_inputs == b.num_inputs) && (a.comparators == b.comparators)
@inline Base.hash(network::SortingNetwork, h::UInt) =
    hash(network.comparators, hash(network.num_inputs, h))


@inline branch_free_minmax(x::T, y::T) where {T} =
    ifelse(x > y, (y, x), (x, y))


function apply_sort!(
    x::AbstractVector,
    network::SortingNetwork,
)
    Base.require_one_based_indexing(x)
    @assert length(x) == network.num_inputs
    for (i, j) in network.comparators
        @inbounds x[i], x[j] = branch_free_minmax(x[i], x[j])
    end
    return x
end


function apply_sort_without!(
    x::AbstractVector,
    network::SortingNetwork,
    index::Int,
)
    Base.require_one_based_indexing(x)
    @assert length(x) == network.num_inputs
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


@inline function two_sum(a::T, b::T) where {T}
    s = a + b
    a_prime = s - b
    b_prime = s - a_prime
    err_a = a - a_prime
    err_b = b - b_prime
    e = err_a + err_b
    return (s, e)
end


function apply_two_sum!(
    x::AbstractVector,
    network::SortingNetwork,
)
    Base.require_one_based_indexing(x)
    @assert length(x) == network.num_inputs
    for (i, j) in network.comparators
        @inbounds x[i], x[j] = two_sum(x[i], x[j])
    end
end


function apply_two_sum_without!(
    x::AbstractVector,
    network::SortingNetwork,
    index::Int,
)
    Base.require_one_based_indexing(x)
    @assert length(x) == network.num_inputs
    @assert 1 <= index <= length(network.comparators)
    for k = 1:index-1
        @inbounds i, j = network.comparators[k]
        @inbounds x[i], x[j] = two_sum(x[i], x[j])
    end
    for k = index+1:length(network.comparators)
        @inbounds i, j = network.comparators[k]
        @inbounds x[i], x[j] = two_sum(x[i], x[j])
    end
end


################################################################################


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


################################################################################


abstract type AbstractCondition end


abstract type AbstractSortingCondition <: AbstractCondition end


abstract type AbstractTwoSumCondition <: AbstractCondition end


################################################################################


export SortedCondition


struct SortedCondition <: AbstractSortingCondition end


@inline (cond::SortedCondition)(x::AbstractVector) = issorted(x)


################################################################################


export WeaklyNormalizedCondition


struct WeaklyNormalizedCondition <: AbstractTwoSumCondition
    num_limbs::Int

    function WeaklyNormalizedCondition(num_limbs::Integer)
        @assert num_limbs >= one(num_limbs)
        return new(Int(num_limbs))
    end
end


@inline function _weakly_normalized(a::T, b::T) where {T}
    if iszero(b)
        return true
    elseif iszero(a)
        return false
    else
        return exponent(a) >= exponent(b) + precision(T)
    end
end


@inline function _weakly_normalized(a::T, b::T, n::Int) where {T}
    if iszero(b)
        return true
    elseif iszero(a)
        return false
    else
        return exponent(a) >= exponent(b) + n * precision(T)
    end
end


function (cond::WeaklyNormalizedCondition)(x::AbstractVector{T}) where {T}
    Base.require_one_based_indexing(x)
    @assert length(x) >= cond.num_limbs
    for i = 1:cond.num_limbs-1
        @inbounds a, b = x[i], x[i+1]
        if !_weakly_normalized(a, b)
            return false
        end
    end
    @inbounds first_limb = x[1]
    for i = cond.num_limbs+1:length(x)
        @inbounds e = x[i]
        if !_weakly_normalized(first_limb, e, cond.num_limbs)
            return false
        end
    end
    return true
end


################################################################################


export StronglyNormalizedCondition


struct StronglyNormalizedCondition <: AbstractTwoSumCondition
    num_limbs::Int

    function StronglyNormalizedCondition(num_limbs::Integer)
        @assert num_limbs >= one(num_limbs)
        return new(Int(num_limbs))
    end
end


function (cond::StronglyNormalizedCondition)(x::AbstractVector)
    Base.require_one_based_indexing(x)
    @assert length(x) >= cond.num_limbs
    for i = 1:cond.num_limbs-1
        @inbounds a, b = x[i], x[i+1]
        if (a, b) !== two_sum(a, b)
            return false
        end
    end
    @inbounds final_limb = x[cond.num_limbs]
    for i = cond.num_limbs+1:length(x)
        @inbounds e = x[i]
        if (final_limb, e) !== two_sum(final_limb, e)
            return false
        end
    end
    return true
end


################################################################################


function satisfies(
    network::SortingNetwork,
    cond::AbstractSortingCondition,
    test_set::Set{Vector{T}},
) where {T}
    v = Vector{T}(undef, network.num_inputs)
    for test_case in test_set
        copyto!(v, test_case)
        apply_sort!(v, network)
        if !cond(v)
            return false
        end
    end
    return true
end


function satisfies(
    network::SortingNetwork,
    cond::AbstractTwoSumCondition,
    test_set::Set{Vector{T}},
) where {T}
    v = Vector{T}(undef, network.num_inputs)
    for test_case in test_set
        copyto!(v, test_case)
        apply_two_sum!(v, network)
        if !cond(v)
            return false
        end
    end
    return true
end


function satisfies_without(
    network::SortingNetwork,
    cond::AbstractSortingCondition,
    test_set::Set{Vector{T}},
    index::Int,
) where {T}
    v = Vector{T}(undef, network.num_inputs)
    for test_case in test_set
        copyto!(v, test_case)
        apply_sort_without!(v, network, index)
        if !cond(v)
            return false
        end
    end
    return true
end


function satisfies_without(
    network::SortingNetwork,
    cond::AbstractTwoSumCondition,
    test_set::Set{Vector{T}},
    index::Int,
) where {T}
    v = Vector{T}(undef, network.num_inputs)
    for test_case in test_set
        copyto!(v, test_case)
        apply_two_sum_without!(v, network, index)
        if !cond(v)
            return false
        end
    end
    return true
end


################################################################################


export generate_sorting_network


function generate_sorting_network(
    cond::AbstractCondition,
    test_set::Set{Vector{T}},
) where {T}

    # Validate input data.
    @assert !isempty(test_set)
    num_inputs = UInt8(length(first(test_set)))
    @assert all(length(test_case) == num_inputs for test_case in test_set)

    # Generate a random sorting network by adding random comparators
    # until the network satisfies the given condition on the test set.
    i_range = Base.OneTo(num_inputs)
    j_range = Base.OneTo(num_inputs - one(UInt8))
    network = SortingNetwork(num_inputs, Tuple{UInt8,UInt8}[])
    while !satisfies(network, cond, test_set)
        i = rand(i_range)
        j = rand(j_range)
        j += (j >= i)
        i, j = branch_free_minmax(i, j)
        push!(network.comparators, (i, j))
    end

    # Prune the network by removing unnecessary comparators.
    while !isempty(network.comparators)
        pruned = false
        for index = 1:length(network.comparators)
            if satisfies_without(network, cond, test_set, index)
                deleteat!(network.comparators, index)
                pruned = true
                break
            end
        end
        if !pruned
            break
        end
    end

    return network
end


################################################################################


abstract type AbstractTestGenerator{T} end


abstract type AbstractSortingTestGenerator{T} <: AbstractTestGenerator{T} end


abstract type AbstractTwoSumTestGenerator{T} <: AbstractTestGenerator{T} end


################################################################################


export search_for_counterexample, search_for_counterexample_timed


function search_for_counterexample(
    network::SortingNetwork,
    cond::AbstractSortingCondition,
    gen::AbstractSortingTestGenerator{T},
    n::Integer,
) where {T}
    v = Vector{T}(undef, network.num_inputs)
    count = zero(n)
    _one = one(n)
    while count < n
        gen(v)
        apply_sort!(v, network)
        if !cond(v)
            return v
        end
        count += _one
    end
    return nothing
end


function search_for_counterexample(
    network::SortingNetwork,
    cond::AbstractTwoSumCondition,
    gen::AbstractTwoSumTestGenerator{T},
    n::Integer,
) where {T}
    v = Vector{T}(undef, network.num_inputs)
    count = zero(n)
    _one = one(n)
    while count < n
        gen(v)
        apply_two_sum!(v, network)
        if !cond(v)
            return v
        end
        count += _one
    end
    return nothing
end


function search_for_counterexample_timed(
    network::SortingNetwork,
    cond::AbstractSortingCondition,
    gen::AbstractSortingTestGenerator{T},
    duration_ns::UInt64,
) where {T}
    start = time_ns()
    stop = start + duration_ns
    v = Vector{T}(undef, network.num_inputs)
    while time_ns() < stop
        gen(v)
        apply_sort!(v, network)
        if !cond(v)
            return v
        end
    end
    return nothing
end


function search_for_counterexample_timed(
    network::SortingNetwork,
    cond::AbstractTwoSumCondition,
    gen::AbstractTwoSumTestGenerator{T},
    duration_ns::UInt64,
) where {T}
    start = time_ns()
    stop = start + duration_ns
    v = Vector{T}(undef, network.num_inputs)
    while time_ns() < stop
        gen(v)
        apply_two_sum!(v, network)
        if !cond(v)
            return v
        end
    end
    return nothing
end


################################################################################


export MultiFloatTestGenerator


struct MultiFloatTestGenerator <: AbstractTwoSumTestGenerator{Float64}
    x::Vector{Float64}
    y::Vector{Float64}

    function MultiFloatTestGenerator(
        num_limbs_x::Integer,
        num_limbs_y::Integer,
    )
        @assert !signbit(num_limbs_x)
        @assert !signbit(num_limbs_y)
        len_x = Int(num_limbs_x)
        len_y = Int(num_limbs_y)
        return new(
            Vector{Float64}(undef, len_x),
            Vector{Float64}(undef, len_y))
    end
end


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


function renormalize!(v::AbstractVector{T}) where {T}
    Base.require_one_based_indexing(v)
    while true
        changed = false
        for i = 1:length(v)-1
            @inbounds x, y = v[i], v[i+1]
            (s, e) = two_sum(x, y)
            changed |= (s !== x) | (e !== y)
            @inbounds v[i], v[i+1] = s, e
        end
        if !changed
            return v
        end
    end
end


function riffle!(
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
    min_len = min(len_x, len_y)
    @simd ivdep for i = 1:min_len
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


function (gen::MultiFloatTestGenerator)(v::AbstractVector{T}) where {T}
    for i in eachindex(gen.x)
        @inbounds gen.x[i] = _generate_random_float()
    end
    renormalize!(gen.x)
    for i in eachindex(gen.y)
        @inbounds gen.y[i] = _generate_random_float()
    end
    renormalize!(gen.y)
    riffle!(v, gen.x, gen.y)
    return v
end


################################################################################


end # module SortingNetworks