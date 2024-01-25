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


end # module ExampleFunctions
