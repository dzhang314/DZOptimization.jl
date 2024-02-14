module Kernels

using MultiFloats: MultiFloat, MultiFloatVec, rsqrt, mfvgather
using SIMD: Vec

###################################################################### UTILITIES

@inline _iota(::Val{M}) where {M} = Vec{M,Int}(ntuple(i -> i - 1, Val{M}()))

#################################################################### DOT PRODUCT

@inline function dot(
    v::AbstractArray{T,D}, w::AbstractArray{T,D}, n::Int
) where {T,D}
    result = zero(T)
    for i = 1:n
        @inbounds result += v[i] * w[i]
    end
    return result
end

@inline function dot_mfv(
    v::Array{MultiFloat{T,N},D}, w::Array{MultiFloat{T,N},D}, n::Int, ::Val{M}
) where {M,T,N,D}
    iota = _iota(Val{M}())
    i = 1
    result_vector = zero(MultiFloatVec{M,T,N})
    while i + M <= n + 1
        result_vector += mfvgather(v, iota + i) * mfvgather(w, iota + i)
        i += M
    end
    result_scalar = zero(MultiFloat{T,N})
    @inbounds while i <= n
        result_scalar += v[i] * w[i]
        i += 1
    end
    return result_scalar + sum(result_vector)
end

# TODO: How do we allow the user to specify the vector length?
# For now, we default to vectors of length 8, since these are fastest on all
# platforms I have tested (Intel 11900KF, AMD Ryzen 9 7950X3D, Apple M3 Pro).

# TODO: Is there a way to vectorize dot products so that the result does not
# depend on the vector length?

################################################################# EUCLIDEAN NORM

@inline function norm2(x::AbstractArray{T,D}, n::Int) where {T,D}
    result = zero(T)
    for i = 1:n
        @inbounds result += abs2(x[i])
    end
    return result
end

@inline function norm2_mfv(x::Array{MultiFloat{T,N},D}, n::Int, ::Val{M}
) where {M,T,N,D}
    iota = _iota(Val{M}())
    i = 1
    result_vector = zero(MultiFloatVec{M,T,N})
    while i + M <= n + 1
        result_vector += abs2(mfvgather(x, iota + i))
        i += M
    end
    result_scalar = zero(MultiFloat{T,N})
    @inbounds while i <= n
        result_scalar += abs2(x[i])
        i += 1
    end
    return result_scalar + sum(result_vector)
end

####################################################################### NEGATION

@inline function negate!(
    x::AbstractArray{T,D}, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds x[i] = -x[i]
    end
    return x
end

########################################################## SCALAR MULTIPLICATION

@inline function scale!(
    x::AbstractArray{T,D}, alpha::T, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds x[i] *= alpha
    end
    return x
end

@inline function scale!(
    dst::AbstractArray{T,D}, alpha::T, x::AbstractArray{T,D}, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds dst[i] = alpha * x[i]
    end
    return x
end

########################################################################## DELTA

@inline function delta!(
    y::AbstractArray{T,D}, x::AbstractArray{T,D}, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds y[i] = x[i] - y[i]
    end
    return y
end

########################################################################### AXPY

@inline function axpy!(
    y::AbstractArray{T,D}, alpha::T, x::AbstractArray{T,D}, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds y[i] += alpha * x[i]
    end
    return y
end

@inline function axpy!(
    dst::AbstractArray{T,D}, alpha::T,
    x::AbstractArray{T,D}, y::AbstractArray{T,D}, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds dst[i] = alpha * x[i] + y[i]
    end
    return dst
end

##################################################################### INTERFACES

@inline norm2(x::Array{T,D}) where {T,D} = norm2(x, length(x))

@inline inv_norm(x::Array{T,D}) where {T,D} = rsqrt(norm2(x))

@inline negate!(x::Array{T,D}) where {T,D} = negate!(x, length(x))

@inline scale!(x::Array{T,D}, alpha::T) where {T,D} =
    scale!(x, alpha, length(x))

end # module Kernels
