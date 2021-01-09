module ExampleFunctions

using ..DZOptimization: unsafe_sqrt


function rosenbrock_objective(v::AbstractVector{T}) where {T}
    @inbounds x, y = v[1], v[2]
    t1 = 1 - x
    t2 = y - x * x
    return t1 * t1 + 100 * (t2 * t2)
end


function rosenbrock_gradient!(g::AbstractVector{T},
                              v::AbstractVector{T}) where {T}
    @inbounds x, y = v[1], v[2]
    t1 = 1 - x
    t2 = y - x * x
    @inbounds g[1] = -2 * t1 - 400 * x * t2
    @inbounds g[2] = 200 * t2
    return g
end


function riesz_energy(points::AbstractMatrix{T}) where {T}
    dim, num_points = size(points)
    energy = zero(T)
    @inbounds for i = 2 : num_points
        for j = 1 : i-1
            dist_sq = zero(T)
            @simd ivdep for k = 1 : dim
                dist = points[k,i] - points[k,j]
                dist_sq += dist * dist
            end
            energy += inv(unsafe_sqrt(dist_sq))
        end
    end
    return energy
end


function riesz_gradient!(
        grad::AbstractMatrix{T}, points::AbstractMatrix{T}) where {T}
    dim, num_points = size(points)
    @inbounds for j = 1 : num_points
        @simd ivdep for k = 1 : dim
            grad[k,j] = zero(T)
        end
        for i = 1 : num_points
            if i != j
                dist_sq = zero(T)
                @simd ivdep for k = 1 : dim
                    dist = points[k,i] - points[k,j]
                    dist_sq += dist * dist
                end
                inv_dist_cubed = unsafe_sqrt(dist_sq) / (dist_sq * dist_sq)
                @simd ivdep for k = 1 : dim
                    grad[k,j] += (points[k,i] - points[k,j]) * inv_dist_cubed
                end
            end
        end
    end
    return grad
end


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


end # module ExampleFunctions
