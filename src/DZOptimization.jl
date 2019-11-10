module DZOptimization

export StepObjectiveFunctor, ConstrainedStepObjectiveFunctor,
    ConstrainedGradientDescentOptimizer, constrained_gradient_descent_optimizer,
    ConstrainedLBFGSOptimizer, constrained_lbfgs_optimizer,
    quadratic_line_search, BFGSOptimizer, step!

using LinearAlgebra: mul!
using DZLinearAlgebra: norm, norm2, dot, identity_matrix!

################################################################################

@inline function _qls_best(fb::T, x1::T, f1::T, x2::T, f2::T,
                           x3::T, f3::T)::Tuple{T,T} where {T<:Real}
    xb = zero(T)
    if f1 < fb; xb, fb = x1, f1; end
    if f2 < fb; xb, fb = x2, f2; end
    if f3 < fb; xb, fb = x3, f3; end
    xb, fb
end

@inline function _qls_minimum_high(f0::T, f1::T, f2::T)::T where {T<:Number}
    q1 = f1 + f1
    q2 = q1 + q1
    q3 = f0 + f0
    q4 = f2 + f0
    q5 = q1 - q4
    (q2 - q3 - q4) / (q5 + q5)
end

@inline function _qls_minimum_low(f0::T, f1::T, f2::T)::T where {T<:Number}
    q1 = f2 + f2
    q2 = q1 + q1
    q3 = f0 + f0
    q4 = f0 + f1
    q5 = q4 - q1
    q6 = q5 + q5
    (q4 + q3 - q2) / (q6 + q6)
end

function quadratic_line_search(f::S, f0::T,
                               x1::T)::Tuple{T,T} where {S,T<:Real}
    # TODO: In principle, we could make this work for f0 == +Inf.
    if !isfinite(f0) || !isfinite(x1)
        return zero(T), f0
    end
    f1 = f(x1)
    while isnan(f1)
        x1 = T(0.5) * x1
        f1 = f(x1)
    end
    if f1 < f0
        while true
            x2 = x1 + x1
            f2 = f(x2)
            if (f2 >= f1) || isnan(f2)
                x3 = x1 * _qls_minimum_high(f0, f1, f2)
                f3 = f(x3)
                return _qls_best(f0, x1, f1, x2, f2, x3, f3)
            else
                x1, f1 = x2, f2
            end
        end
    else
        while true
            x2 = T(0.5) * x1
            f2 = f(x2)
            if isnan(f2)
                return zero(T), f0
            end
            if f2 <= f0
                x3 = x1 * _qls_minimum_low(f0, f1, f2)
                f3 = f(x3)
                return _qls_best(f0, x2, f2, x1, f1, x3, f3)
            else
                x1, f1 = x2, f2
            end
        end
    end
end

################################################################################

function update_inverse_hessian!(B_inv::Matrix{T}, h::T, s::Vector{T},
        y::Vector{T}, t::Vector{T})::Nothing where {T<:Real}
    b = dot(s, y)
    s .*= inv(b)
    mul!(t, B_inv, y)
    a = h * b + dot(y, t)
    n = size(B_inv, 1)
    @inbounds for j = 1 : n
        sj = s[j]
        tj = t[j]
        @simd ivdep for i = 1 : n
            B_inv[i, j] += a * (s[i] * sj) - (t[i] * sj + s[i] * tj)
        end
    end
end

################################################################################

struct StepObjectiveFunctor{S,T<:Real,N}
    objective_functor::S
    initial_point::Array{T,N}
    new_point::Array{T,N}
    step_direction::Array{T,N}
end

struct ConstrainedStepObjectiveFunctor{S1,S2,T<:Real,N}
    objective_functor::S1
    constraint_functor!::S2
    initial_point::Array{T,N}
    new_point::Array{T,N}
    step_direction::Array{T,N}
end

@inline function (so::StepObjectiveFunctor{S,T})(
        step_size::T) where {S,T<:Real}
    x0, x1, dx = so.initial_point, so.new_point, so.step_direction
    @simd ivdep for i = 1 : length(x0)
        @inbounds x1[i] = x0[i] - step_size * dx[i]
    end
    return so.objective_functor(x1)
end

@inline function (cso::ConstrainedStepObjectiveFunctor{S1,S2,T,N})(
        step_size::T) where {S1,S2,T<:Real,N}
    x0, x1, dx = cso.initial_point, cso.new_point, cso.step_direction
    @simd ivdep for i = 1 : length(x0)
        @inbounds x1[i] = x0[i] - step_size * dx[i]
    end
    cso.constraint_functor!(x1)
    return cso.objective_functor(x1)
end

################################################################################

struct ConstrainedGradientDescentOptimizer{S1,S2,S3,T<:Real,N}
    objective_functor::S1
    gradient_functor!::S2
    constraint_functor!::S3
    current_iteration::Vector{Int}
    current_objective::Vector{T}
    current_point::Array{T,N}
    current_gradient::Array{T,N}
    delta_point::Array{T,N}
    delta_gradient::Array{T,N}
    scratch_space::Array{T,N}
    step_functor::ConstrainedStepObjectiveFunctor{S1,S3,T,N}
end

function constrained_gradient_descent_optimizer(
        objective_functor::S1, gradient_functor!::S2, constraint_functor!::S3,
        initial_point::AbstractArray{T,N}) where {S1,S2,S3,T<:Real,N}
    current_point = copy(initial_point)
    constraint_functor!(current_point)
    current_objective = objective_functor(current_point)
    current_gradient = similar(current_point)
    gradient_functor!(current_gradient, current_point)
    delta_point = zero(current_point)
    delta_gradient = zero(current_gradient)
    scratch_space = similar(current_point)
    step_functor = ConstrainedStepObjectiveFunctor{S1,S3,T,N}(
        objective_functor, constraint_functor!,
        current_point, scratch_space, current_gradient)
    return ConstrainedGradientDescentOptimizer{S1,S2,S3,T,N}(
        objective_functor, gradient_functor!, constraint_functor!, Int[0],
        T[current_objective], current_point, current_gradient,
        delta_point, delta_gradient, scratch_space, step_functor)
end

function step!(opt::ConstrainedGradientDescentOptimizer{S1,S2,S3,T,N}
        ) where {S1,S2,S3,T<:Real,N}
    @inbounds begin
        x, g = opt.current_point, opt.current_gradient
        s, y = opt.delta_point, opt.delta_gradient
        n, temp = length(x), opt.scratch_space
        initial_step_size = dot(s, y) / norm2(y)
        if !isfinite(initial_step_size)
            initial_step_size = sqrt(eps(T)) / max(one(T), norm2(g))
        end
        step_size, new_objective = quadratic_line_search(
            opt.step_functor, opt.current_objective[1], initial_step_size)
        opt.current_iteration[1] += 1
        opt.current_objective[1] = new_objective
        @simd ivdep for i = 1:n; temp[i] = x[i] - step_size * g[i]; end
        opt.constraint_functor!(temp)
        @simd ivdep for i = 1:n; s[i] = temp[i] - x[i]            ; end
        @simd ivdep for i = 1:n; x[i] = temp[i]                   ; end
        opt.gradient_functor!(temp, x)
        @simd ivdep for i = 1:n; y[i] = temp[i] - g[i]            ; end
        @simd ivdep for i = 1:n; g[i] = temp[i]                   ; end
    end
    return opt
end

################################################################################

struct ConstrainedLBFGSOptimizer{S1,S2,S3,T<:Real,N}
    objective_functor::S1
    gradient_functor!::S2
    constraint_functor!::S3
    current_iteration::Vector{Int}
    current_objective::Vector{T}
    current_point::Array{T,N}
    current_gradient::Array{T,N}
    delta_point::Array{T,N}
    delta_gradient::Array{T,N}
    delta_point_history::Matrix{T}
    delta_gradient_history::Matrix{T}
    alpha_history::Vector{T}
    rho_history::Vector{T}
    history_length::Int
    step_direction::Array{T,N}
    scratch_space::Array{T,N}
    step_functor::ConstrainedStepObjectiveFunctor{S1,S3,T,N}
end

function constrained_lbfgs_optimizer(
        objective_functor::S1, gradient_functor!::S2, constraint_functor!::S3,
        initial_point::AbstractArray{T,N}, m::Int) where {S1,S2,S3,T<:Real,N}
    current_point = copy(initial_point)
    constraint_functor!(current_point)
    current_objective = objective_functor(current_point)
    current_gradient = similar(current_point)
    gradient_functor!(current_gradient, current_point)
    delta_point = zero(current_point)
    delta_gradient = zero(current_gradient)
    step_direction = similar(current_point)
    scratch_space = similar(current_point)
    step_functor = ConstrainedStepObjectiveFunctor{S1,S3,T,N}(
        objective_functor, constraint_functor!,
        current_point, scratch_space, step_direction)
    return ConstrainedLBFGSOptimizer{S1,S2,S3,T,N}(
        objective_functor, gradient_functor!, constraint_functor!, Int[0],
        T[current_objective], current_point, current_gradient,
        delta_point, delta_gradient,
        Matrix{T}(undef, length(current_point), m),
        Matrix{T}(undef, length(current_point), m),
        Vector{T}(undef, m), Vector{T}(undef, m),
        m, step_direction, scratch_space, step_functor)
end

function step!(opt::ConstrainedLBFGSOptimizer{S1,S2,S3,T,N}
        ) where {S1,S2,S3,T<:Real,N}
    @inbounds begin

        # Define aliases to opt member variables
        x, g = opt.current_point, opt.current_gradient
        s, y = opt.delta_point, opt.delta_gradient
        S, Y = opt.delta_point_history, opt.delta_gradient_history
        alpha, rho = opt.alpha_history, opt.rho_history
        m, n, temp = opt.history_length, length(x), opt.scratch_space
        q, k = opt.step_direction, opt.current_iteration[1] + 1
        cur_objective = opt.current_objective[1]

        # Compute step direction
        @simd ivdep for j = 1:n; q[j] = g[j]                      ; end
        for i = k-1 : -1 : max(k-m, 1)
            c = (i - 1) % m + 1
            d = zero(T)
            @simd ivdep for j = 1:n; d += S[j,c] * q[j]           ; end
            a = alpha[c] = rho[c] * d
            @simd ivdep for j = 1:n; q[j] -= a * Y[j,c]           ; end
        end
        gamma = dot(s, y) / norm2(y)
        if !isfinite(gamma)
            gamma = sqrt(eps(T)) / max(one(T), norm2(g))
        end
        @simd ivdep for j = 1:n; q[j] *= gamma; end
        for i = max(k-m, 1) : k-1
            c = (i - 1) % m + 1
            d = zero(T)
            @simd ivdep for j = 1:n; d += Y[j,c] * q[j]           ; end
            beta = alpha[c] - rho[c] * d
            @simd ivdep for j = 1:n; q[j] += S[j,c] * beta        ; end
        end

        # Perform line search
        step_size, new_objective = quadratic_line_search(
            opt.step_functor, cur_objective, one(T))

        # Did we improve? If not, return early
        if !(new_objective < cur_objective)
            return false
        end

        # If we did improve, accept the step
        opt.current_iteration[1] = k
        opt.current_objective[1] = new_objective
        @simd ivdep for j = 1:n; temp[j] = x[j] - step_size * q[j]; end
        opt.constraint_functor!(temp)
        @simd ivdep for j = 1:n; s[j] = temp[j] - x[j]            ; end
        @simd ivdep for j = 1:n; x[j] = temp[j]                   ; end
        opt.gradient_functor!(temp, x)
        @simd ivdep for j = 1:n; y[j] = temp[j] - g[j]            ; end
        @simd ivdep for j = 1:n; g[j] = temp[j]                   ; end
        c = (k - 1) % m + 1
        @simd ivdep for j = 1:n; S[j,c] = s[j]                    ; end
        @simd ivdep for j = 1:n; Y[j,c] = y[j]                    ; end
        rho[c] = inv(dot(s, y))
    end
    return true
end

################################################################################

struct BFGSOptimizer{S1, S2, T <: Real}
    num_dims::Int
    iteration::Vector{Int}
    objective_functor::S1
    gradient_functor!::S2
    current_point::Vector{T}
    temp_buffer::Vector{T}
    objective::Vector{T}
    gradient::Vector{T}
    last_step_size::Vector{T}
    delta_gradient::Vector{T}
    bfgs_dir::Vector{T}
    hess_inv::Matrix{T}
    grad_functor::StepObjectiveFunctor{S1,T,1}
    bfgs_functor::StepObjectiveFunctor{S1,T,1}
end

function BFGSOptimizer(initial_point::Vector{T}, initial_step_size::T,
        objective_functor::S1, gradient_functor!::S2) where {S1, S2, T <: Real}
    num_dims = length(initial_point)
    current_point = copy(initial_point)
    temp_buffer = Vector{T}(undef, num_dims)
    objective = objective_functor(current_point)
    gradient = Vector{T}(undef, num_dims)
    gradient_functor!(gradient, current_point)
    bfgs_dir = copy(gradient)
    hess_inv = Matrix{T}(undef, num_dims, num_dims)
    identity_matrix!(hess_inv)
    BFGSOptimizer{S1,S2,T}(num_dims, Int[0],
        objective_functor, gradient_functor!,
        current_point, temp_buffer, T[objective], gradient,
        T[initial_step_size], Vector{T}(undef, num_dims),
        bfgs_dir, hess_inv,
        StepObjectiveFunctor{S1,T,1}(objective_functor, current_point,
            temp_buffer, gradient),
        StepObjectiveFunctor{S1,T,1}(objective_functor, current_point,
            temp_buffer, bfgs_dir))
end

function step!(opt::BFGSOptimizer{S1,S2,T}) where {S1, S2, T <: Real}
    @inbounds step_size, objective = opt.last_step_size[1], opt.objective[1]
    grad_dir, bfgs_dir = opt.gradient, opt.bfgs_dir
    delta_grad, hess_inv = opt.delta_gradient, opt.hess_inv
    x, n = opt.current_point, opt.num_dims
    grad_norm, bfgs_norm = norm(grad_dir), norm(bfgs_dir)
    grad_step_size, grad_obj = quadratic_line_search(
        opt.grad_functor, objective, step_size / grad_norm)
    bfgs_step_size, bfgs_obj = quadratic_line_search(
        opt.bfgs_functor, objective, step_size / bfgs_norm)
    if bfgs_obj <= grad_obj
        if !(bfgs_obj < objective)
            return true, false
        end
        @inbounds opt.objective[1] = bfgs_obj
        @inbounds opt.last_step_size[1] = bfgs_step_size * bfgs_norm
        @simd ivdep for i = 1 : n
            @inbounds x[i] -= bfgs_step_size * bfgs_dir[i]
        end
        @simd ivdep for i = 1 : n
            @inbounds delta_grad[i] = -grad_dir[i]
        end
        opt.gradient_functor!(grad_dir, x)
        @simd ivdep for i = 1 : n
            @inbounds delta_grad[i] += grad_dir[i]
        end
        update_inverse_hessian!(hess_inv, -bfgs_step_size, bfgs_dir,
            delta_grad, opt.temp_buffer)
        mul!(bfgs_dir, hess_inv, grad_dir)
        @inbounds opt.iteration[1] += 1
        return true, true
    else
        if !(grad_obj < objective)
            return false, false
        end
        @inbounds opt.objective[1] = grad_obj
        @inbounds opt.last_step_size[1] = grad_step_size * grad_norm
        @simd ivdep for i = 1 : n
            @inbounds x[i] -= grad_step_size * grad_dir[i]
        end
        identity_matrix!(hess_inv)
        opt.gradient_functor!(grad_dir, x)
        @simd ivdep for i = 1 : n
            @inbounds bfgs_dir[i] = grad_dir[i]
        end
        @inbounds opt.iteration[1] += 1
        return false, true
    end
end

end # module DZOptimization
