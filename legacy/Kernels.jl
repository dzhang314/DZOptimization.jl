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

@inline norm2(x::AbstractArray{T,D}) where {T,D} = norm2(x, length(x))

@inline inv_norm(x::AbstractArray{T,D}) where {T,D} = rsqrt(norm2(x))

@inline negate!(x::AbstractArray{T,D}) where {T,D} = negate!(x, length(x))

@inline scale!(x::AbstractArray{T,D}, alpha::T) where {T,D} =
    scale!(x, alpha, length(x))

############################################################## ORTHOGONALIZATION

function orthogonalize_columns!(A::Matrix{T}) where {T}
    n, m = size(A)
    @inbounds for i = 1:m
        A_i = view(A, :, i)
        squared_norm = norm2(A_i, n)
        if !iszero(squared_norm)
            inv_norm = rsqrt(squared_norm)
            # diagonal entry of R is inv(inv_norm)
            scale!(A_i, inv_norm, n)
            for j = i+1:m
                A_j = view(A, :, j)
                overlap = dot(A_i, A_j, n)
                # row of R is overlap
                axpy!(A_j, -overlap, A_i, n)
            end
        end
    end
    return A
end

function stratified_orthogonal_basis!(matrices::Vector{Matrix{T}}) where {T}
    _eps = eps(T)
    _eps_1_2 = sqrt(_eps)
    _eps_1_4 = sqrt(_eps_1_2)
    _eps_1_8 = sqrt(_eps_1_4)
    _eps_loose = _eps_1_4 * _eps_1_8    # 3/8ths of digits
    _eps_strict = _eps_1_2 * _eps_loose # 7/8ths of digits
    _eps_strict_sq = _eps_strict * _eps_strict
    result = Tuple{Int,Int,Vector{T}}[]
    for (i, matrix_i) in enumerate(matrices)
        m, n = size(matrix_i)
        if iszero(n)
            continue
        elseif isone(n)
            squared_norm = norm2(view(matrix_i, :, 1))
            if squared_norm < _eps_strict_sq
                break
            elseif squared_norm < _eps_loose
                return nothing
            else
                column_one = matrix_i[:, 1]
                push!(result, (i, 1, column_one))
                for j = i:length(matrices)
                    matrix_j = matrices[j]
                    @assert m == size(matrix_j, 1)
                    for k = 1:size(matrix_j, 2)
                        column_k = view(matrix_j, :, k)
                        overlap = dot(column_one, column_k, m)
                        axpy!(column_k, -overlap / squared_norm, column_one, m)
                    end
                end
                continue
            end
        end
        while true
            squared_norms = [(norm2(view(matrix_i, :, j)), j) for j = 1:n]
            sort!(squared_norms) # TODO: don't need a full sort
            max_squared_norm, max_index = squared_norms[end]
            next_squared_norm, _ = squared_norms[end-1]
            if max_squared_norm < _eps_strict_sq
                break
            elseif max_squared_norm < _eps_loose
                return nothing
            elseif max_squared_norm - next_squared_norm < _eps_loose
                return nothing
            end
            column_max = matrix_i[:, max_index]
            push!(result, (i, max_index, column_max))
            for j = i:length(matrices)
                matrix_j = matrices[j]
                @assert m == size(matrix_j, 1)
                for k = 1:size(matrix_j, 2)
                    column_k = view(matrix_j, :, k)
                    overlap = dot(column_max, column_k, m)
                    axpy!(column_k, -overlap / max_squared_norm, column_max, m)
                end
            end
        end
    end
    return result
end

end # module Kernels
