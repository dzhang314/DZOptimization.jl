module Kernels

using MultiFloats: rsqrt
using SIMD: Vec

###################################################################### UTILITIES

@inline _iota(::Val{M}) where {M} = Vec{M,Int}(ntuple(i -> i - 1, Val{M}()))

#################################################################### DOT PRODUCT

@inline function dot(
    v::Array{T,D}, w::Array{T,D}, n::Int
) where {T,D}
    result = zero(T)
    @simd for i = 1:n
        @inbounds result += v[i] * w[i]
    end
    return result
end

@inline function dot_column(
    v::Array{T,D}, w::Matrix{T}, j::Int, n::Int
) where {T,D}
    result = zero(T)
    @simd for i = 1:n
        @inbounds result += v[i] * w[i, j]
    end
    return result
end

################################################################# EUCLIDEAN NORM

@inline function norm2(x::Array{T,D}, n::Int) where {T,D}
    result = zero(T)
    @simd for i = 1:n
        @inbounds result += abs2(x[i])
    end
    return result
end

# function norm2_mfv(x::Array{MultiFloat{T,N},D}, ::Val{M}) where {M,T,N,D}
#     n = length(x)
#     iota = _iota(Val{M}())
#     i = 1
#     result_vector = zero(MultiFloatVec{M,T,N})
#     while i + M <= n + 1
#         result_vector += abs2(mfvgather(x, iota + i))
#         i += M
#     end
#     result_scalar = zero(MultiFloat{T,N})
#     @inbounds while i <= n
#         result_scalar += abs2(x[i])
#         i += 1
#     end
#     return result_scalar + sum(result_vector)
# end

# # TODO: How do we allow the user to specify the vector length?
# # For now, we default to vectors of length 8, since these are fastest on all
# # platforms I have tested (Intel 11900KF, AMD Ryzen 9 7950X3D, Apple M3 Pro).
# @inline norm2(x::Array{MultiFloat{T,N},D}) where {T,N,D} =
#     norm2_mfv(x, Val{8}())

####################################################################### NEGATION

@inline function negate!(
    x::Array{T,D}, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds x[i] = -x[i]
    end
    return x
end

########################################################## SCALAR MULTIPLICATION

@inline function scale!(
    x::Array{T,D}, alpha::T, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds x[i] *= alpha
    end
    return x
end

@inline function scale!(
    dst::Array{T,D}, alpha::T, x::Array{T,D}, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds dst[i] = alpha * x[i]
    end
    return x
end

########################################################################## DELTA

@inline function delta!(
    y::Array{T,D}, x::Array{T,D}, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds y[i] = x[i] - y[i]
    end
    return y
end

########################################################################### AXPY

@inline function axpy!(
    y::Array{T,D}, alpha::T, x::Array{T,D}, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds y[i] += alpha * x[i]
    end
    return y
end

@inline function axpy!(
    dst::Array{T,D}, alpha::T, x::Array{T,D}, y::Array{T,D}, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds dst[i] = alpha * x[i] + y[i]
    end
    return dst
end

@inline function axpy_column!(
    y::Array{T,D}, alpha::T, x::Matrix{T}, j::Int, n::Int
) where {T,D}
    @simd ivdep for i = 1:n
        @inbounds y[i] += alpha * x[i, j]
    end
    return y
end

##################################################################### INTERFACES

@inline norm2(x::Array{T,D}) where {T,D} = norm2(x, length(x))

@inline inv_norm(x::Array{T,D}) where {T,D} = rsqrt(norm2(x))

@inline negate!(x::Array{T,D}) where {T,D} = negate!(x, length(x))

@inline scale!(x::Array{T,D}, alpha::T) where {T,D} =
    scale!(x, alpha, length(x))

end # module Kernels
