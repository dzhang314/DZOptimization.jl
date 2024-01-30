module ExampleFunctions

using MultiFloats: MultiFloat, MultiFloatVec, rsqrt, mfvgather
using SIMD: Vec

##################################################################### ROSENBROCK

export rosenbrock_function, rosenbrock_gradient!

function rosenbrock_function(v::Vector{T}) where {T}
    @inbounds x, y = v[1], v[2]
    t1 = 1 - x
    t2 = y - x * x
    return t1 * t1 + 100 * (t2 * t2)
end

function rosenbrock_gradient!(g::Vector{T}, v::Vector{T}) where {T}
    @inbounds x, y = v[1], v[2]
    t1 = 1 - x
    t2 = y - x * x
    @inbounds g[1] = -2 * t1 - 400 * x * t2
    @inbounds g[2] = 200 * t2
    return g
end

################################################################### RIESZ ENERGY

export riesz_energy, riesz_gradient!, riesz_gradient

function riesz_energy(points::Matrix{T}) where {T}
    dimension, num_points = size(points)
    result = zero(T)
    @inbounds for j = 2:num_points
        for i = 1:j-1
            dist_sq = zero(T)
            @simd for k = 1:dimension
                dist = points[k, i] - points[k, j]
                dist_sq += dist * dist
            end
            result += rsqrt(dist_sq)
        end
    end
    return result
end

function riesz_gradient!(gradient::Matrix{T}, points::Matrix{T}) where {T}
    dimension, num_points = size(points)
    @assert (dimension, num_points) == size(gradient)
    @inbounds for j = 1:num_points
        @simd ivdep for k = 1:dimension
            gradient[k, j] = zero(T)
        end
        for i = 1:j-1
            dist_sq = zero(T)
            @simd for k = 1:dimension
                dist = points[k, i] - points[k, j]
                dist_sq += dist * dist
            end
            inv_dist = rsqrt(dist_sq)
            inv_dist_cubed = inv_dist / dist_sq
            @simd ivdep for k = 1:dimension
                dist = points[k, i] - points[k, j]
                gradient[k, j] += dist * inv_dist_cubed
            end
        end
        for i = j+1:num_points
            dist_sq = zero(T)
            @simd for k = 1:dimension
                dist = points[k, i] - points[k, j]
                dist_sq += dist * dist
            end
            inv_dist = rsqrt(dist_sq)
            inv_dist_cubed = inv_dist / dist_sq
            @simd ivdep for k = 1:dimension
                dist = points[k, i] - points[k, j]
                gradient[k, j] += dist * inv_dist_cubed
            end
        end
    end
    return gradient
end

function riesz_gradient(points::Matrix{T}) where {T}
    result = similar(points)
    riesz_gradient!(result, points)
    return result
end

################################################################ RIESZ ENERGY 2D

export riesz_energy_2d, riesz_gradient_2d!, riesz_gradient_2d

function riesz_energy_2d(points::Matrix{T}) where {T}
    dimension, num_points = size(points)
    @assert dimension == 2
    result = zero(T)
    @inbounds for j = 2:num_points
        xj = points[1, j]
        yj = points[2, j]
        @simd for i = 1:j-1
            dx = points[1, i] - xj
            dy = points[2, i] - yj
            result += rsqrt(dx * dx + dy * dy)
        end
    end
    return result
end

function riesz_gradient_2d!(gradient::Matrix{T}, points::Matrix{T}) where {T}
    dimension, num_points = size(points)
    @assert (dimension, num_points) == size(gradient)
    @assert dimension == 2
    @inbounds for j = 1:num_points
        gx = zero(T)
        gy = zero(T)
        xj = points[1, j]
        yj = points[2, j]
        for i = 1:j-1
            dx = points[1, i] - xj
            dy = points[2, i] - yj
            dist_sq = dx * dx + dy * dy
            inv_dist = rsqrt(dist_sq)
            inv_dist_cubed = inv_dist / dist_sq
            gx += dx * inv_dist_cubed
            gy += dy * inv_dist_cubed
        end
        for i = j+1:num_points
            dx = points[1, i] - xj
            dy = points[2, i] - yj
            dist_sq = dx * dx + dy * dy
            inv_dist = rsqrt(dist_sq)
            inv_dist_cubed = inv_dist / dist_sq
            gx += dx * inv_dist_cubed
            gy += dy * inv_dist_cubed
        end
        gradient[1, j] = gx
        gradient[2, j] = gy
    end
    return gradient
end

function riesz_gradient_2d(points::Matrix{T}) where {T}
    result = similar(points)
    riesz_gradient_2d!(result, points)
    return result
end

######################################## RIESZ ENERGY 2D (MULTIFLOAT-VECTORIZED)

export riesz_energy_2d_mfv, riesz_gradient_2d_mfv!, riesz_gradient_2d_mfv

@inline _iota(::Val{M}) where {M} = Vec{M,Int}(ntuple(i -> i - 1, Val{M}()))

function riesz_energy_2d_mfv(
    points::Matrix{MultiFloat{T,N}},
    ::Val{M}
) where {M,T,N}
    dimension, num_points = size(points)
    @assert dimension == 2
    ptr = pointer(points)
    iota_x = 2 * (_iota(Val{M}()) - 1)
    iota_y = iota_x + 1
    result_scalar = zero(MultiFloat{T,N})
    result_vector = zero(MultiFloatVec{M,T,N})
    @inbounds for j = 2:num_points
        xj = points[1, j]
        yj = points[2, j]
        i = 1
        while i + M <= j
            twice_i = i + i
            xi = mfvgather(ptr, iota_x + twice_i)
            yi = mfvgather(ptr, iota_y + twice_i)
            dx = xi - xj
            dy = yi - yj
            result_vector += rsqrt(dx * dx + dy * dy)
            i += M
        end
        while i < j
            xi = points[1, i]
            yi = points[2, i]
            dx = xi - xj
            dy = yi - yj
            result_scalar += rsqrt(dx * dx + dy * dy)
            i += 1
        end
    end
    return result_scalar + sum(result_vector)
end

@inline riesz_energy_2d_mfv(points::Matrix{MultiFloat{T,N}}) where {T,N} =
    riesz_energy_2d_mfv(points, Val{8}())

function riesz_gradient_2d_mfv!(
    gradient::Matrix{MultiFloat{T,N}},
    points::Matrix{MultiFloat{T,N}},
    ::Val{M}
) where {M,T,N}
    dimension, num_points = size(points)
    @assert (dimension, num_points) == size(gradient)
    @assert dimension == 2
    ptr = pointer(points)
    iota_x = 2 * (_iota(Val{M}()) - 1)
    iota_y = iota_x + 1
    @inbounds for j = 1:num_points
        gx_scalar = zero(MultiFloat{T,N})
        gy_scalar = zero(MultiFloat{T,N})
        gx_vector = zero(MultiFloatVec{M,T,N})
        gy_vector = zero(MultiFloatVec{M,T,N})
        xj = points[1, j]
        yj = points[2, j]
        i = 1
        while i + M <= j
            twice_i = i + i
            xi = mfvgather(ptr, iota_x + twice_i)
            yi = mfvgather(ptr, iota_y + twice_i)
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            inv_dist = rsqrt(dist_sq)
            inv_dist_cubed = inv_dist / dist_sq
            gx_vector += dx * inv_dist_cubed
            gy_vector += dy * inv_dist_cubed
            i += M
        end
        while i < j
            xi = points[1, i]
            yi = points[2, i]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            inv_dist = rsqrt(dist_sq)
            inv_dist_cubed = inv_dist / dist_sq
            gx_scalar += dx * inv_dist_cubed
            gy_scalar += dy * inv_dist_cubed
            i += 1
        end
        i = j + 1
        while i + M <= num_points + 1
            twice_i = i + i
            xi = mfvgather(ptr, iota_x + twice_i)
            yi = mfvgather(ptr, iota_y + twice_i)
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            inv_dist = rsqrt(dist_sq)
            inv_dist_cubed = inv_dist / dist_sq
            gx_vector += dx * inv_dist_cubed
            gy_vector += dy * inv_dist_cubed
            i += M
        end
        while i <= num_points
            xi = points[1, i]
            yi = points[2, i]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            inv_dist = rsqrt(dist_sq)
            inv_dist_cubed = inv_dist / dist_sq
            gx_scalar += dx * inv_dist_cubed
            gy_scalar += dy * inv_dist_cubed
            i += 1
        end
        gradient[1, j] = gx_scalar + sum(gx_vector)
        gradient[2, j] = gy_scalar + sum(gy_vector)
    end
    return gradient
end

@inline riesz_gradient_2d_mfv!(
    gradient::Matrix{MultiFloat{T,N}}, points::Matrix{MultiFloat{T,N}}
) where {T,N} = riesz_gradient_2d_mfv!(gradient, points, Val{8}())

function riesz_gradient_2d_mfv(
    points::Matrix{MultiFloat{T,N}},
    ::Val{M}
) where {M,T,N}
    result = similar(points)
    riesz_gradient_2d_mfv!(result, points, Val{M}())
    return result
end

@inline riesz_gradient_2d_mfv(points::Matrix{MultiFloat{T,N}}) where {T,N} =
    riesz_gradient_2d_mfv(points, Val{8}())

############################################################# FINITE DIFFERENCES

export finite_difference_gradient

function finite_difference_gradient(f, x::Array{T,N}, h::T) where {T,N}
    g = similar(x)
    inv_twice_h = inv(h + h)
    for i in eachindex(x)
        xi_old = x[i]
        x[i] = xi_old + h
        f_pos = f(x)
        x[i] = xi_old - h
        f_neg = f(x)
        x[i] = xi_old
        g[i] = (f_pos - f_neg) * inv_twice_h
    end
    return g
end

#=

function riesz_hessian!(
        hess::AbstractArray{T,4}, points::AbstractMatrix{T}) where {T}
    dim, num_points = size(points)
    @inbounds for k = 1 : num_points
        for s = 1 : num_points
            if s == k
                for l = 1 : dim
                    @simd ivdep for t = 1 : dim
                        hess[t,s,l,k] = zero(T)
                    end
                end
                for j = 1 : num_points
                    if j != k
                        dist_sq = zero(T)
                        @simd ivdep for d = 1 : dim
                            temp = points[d,k] - points[d,j]
                            dist_sq += temp * temp
                        end
                        dist = unsafe_sqrt(dist_sq)
                        dist_cb = dist * dist_sq
                        for l = 1 : dim
                            @simd ivdep for t = 1 : dim
                                plkj = points[l,k] - points[l,j]
                                ptkj = points[t,k] - points[t,j]
                                temp = (plkj * ptkj) / (dist_sq * dist_cb)
                                hess[t,s,l,k] += (temp + temp + temp)
                            end
                            hess[l,s,l,k] -= inv(dist_cb)
                        end
                    end
                end
            else
                dist_sq = zero(T)
                @simd ivdep for d = 1 : dim
                    temp = points[d,k] - points[d,s]
                    dist_sq += temp * temp
                end
                dist = unsafe_sqrt(dist_sq)
                dist_cb = dist * dist_sq
                for l = 1 : dim
                    @simd ivdep for t = 1 : dim
                        plks = points[l,k] - points[l,s]
                        ptsk = points[t,s] - points[t,k]
                        temp = (plks * ptsk) / (dist_sq * dist_cb)
                        hess[t,s,l,k] = (temp + temp + temp)
                    end
                    hess[l,s,l,k] += inv(dist_cb)
                end
            end
        end
    end
    return hess
end

function constrain_riesz_gradient_sphere!(
        grad::AbstractMatrix{T}, points::AbstractMatrix{T}) where {T}
    dim, num_points = size(points)
    @inbounds for i = 1 : num_points
        overlap = zero(T)
        @simd ivdep for k = 1 : dim
            overlap += points[k,i] * grad[k,i]
        end
        @simd ivdep for k = 1 : dim
            grad[k,i] -= overlap * points[k,i]
        end
    end
    return grad
end

function constrain_riesz_hessian_sphere!(
        hess::AbstractArray{T,4}, points::AbstractMatrix{T},
        unconstrained_grad::AbstractMatrix{T}) where {T}
    dim, num_points = size(points)
    @inbounds for s = 1 : num_points
        for t = 1 : num_points
            for i = 1 : dim
                temp = zero(T)
                @simd ivdep for j = 1 : dim
                    temp += hess[j,s,i,t] * points[j,s]
                end
                @simd ivdep for j = 1 : dim
                    hess[j,s,i,t] -= temp * points[j,s]
                end
            end
        end
        for t = 1 : num_points
            for i = 1 : dim
                temp = zero(T)
                @simd ivdep for j = 1 : dim
                    temp += hess[i,t,j,s] * points[j,s]
                end
                @simd ivdep for j = 1 : dim
                    hess[i,t,j,s] -= temp * points[j,s]
                end
            end
        end
    end
    @inbounds for s = 1 : num_points
        for i = 1 : dim
            gis = unconstrained_grad[i,s]
            pis = points[i,s]
            for j = 1 : dim
                pjs = points[j,s]
                for k = 1 : dim
                    pks = points[k,s]
                    temp = pis * pjs * pks
                    temp = temp + temp + temp
                    if j == k; temp -= pis; end
                    if i == k; temp -= pjs; end
                    if i == j; temp -= pks; end
                    hess[k,s,j,s] += gis * temp
                end
            end
        end
    end
    return hess
end

=#

end # module ExampleFunctions
