module PCG


export random_fill!, random_array


@inline _pcg_advance(state::UInt64) =
    0x5851F42D4C957F2D * state + 0x14057B7EF767814F


@inline _pcg_extract(state::UInt64) = bitrotate(
    (xor(state >> 18, state) >> 27) % UInt32, -(state >> 59))


@inline function random_fill!(x::AbstractArray, seed::Integer)
    state = _pcg_advance(0x14057B7EF767814F + (seed % UInt64))
    for i in eachindex(x)
        @inbounds x[i] = 2.3283064365386962890625E-10 * _pcg_extract(state)
        state = _pcg_advance(state)
    end
    return x
end


@inline random_array(
    seed::Integer,
    ::Type{T},
    dims::NTuple{D,Int},
) where {T,D} = random_fill!(Array{T,D}(undef, dims), seed % UInt64)


@inline random_array(seed::Integer, ::Type{T}, dims::Int...) where {T} =
    random_array(seed, T, dims)


end # module PCG
