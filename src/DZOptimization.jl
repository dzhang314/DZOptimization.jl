module DZOptimization

export LineSearchFunctor, ConstrainedLineSearchFunctor,
    ConstrainedGradientDescentOptimizer, constrained_gradient_descent_optimizer,
    ConstrainedLBFGSOptimizer, constrained_lbfgs_optimizer,
    quadratic_line_search, BFGSOptimizer, step!

using LinearAlgebra: mul!
using DZLinearAlgebra: norm, norm2, dot, identity_matrix!

######################################################### LINE SEARCH ALGORITHMS

@inline function _qls_best(fb::T, x1::T, f1::T,
                           x2::T, f2::T, x3::T, f3::T) where {T}
    xb = zero(T)
    if f1 < fb; xb, fb = x1, f1; end
    if f2 < fb; xb, fb = x2, f2; end
    if f3 < fb; xb, fb = x3, f3; end
    return (xb, fb)
end

@inline function _qls_minimum_high(f0::T, f1::T, f2::T) where {T}
    q1 = f1 + f1
    q2 = q1 + q1
    q3 = f0 + f0
    q4 = f2 + f0
    q5 = q1 - q4
    return (q2 - q3 - q4) / (q5 + q5)
end

@inline function _qls_minimum_low(f0::T, f1::T, f2::T)::T where {T}
    q1 = f2 + f2
    q2 = q1 + q1
    q3 = f0 + f0
    q4 = f0 + f1
    q5 = q4 - q1
    q6 = q5 + q5
    return (q4 + q3 - q2) / (q6 + q6)
end

function quadratic_line_search(f::F, f0::T, x1::T) where {F,T}
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
                return (zero(T), f0)
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

################################################### LINE SEARCH FUNCTION OBJECTS

struct LineSearchFunctor{S,T<:Real,N}
    objective_functor::S
    initial_point::Array{T,N}
    new_point::Array{T,N}
    step_direction::Array{T,N}
end

struct ConstrainedLineSearchFunctor{S1,S2,T<:Real,N}
    objective_functor::S1
    constraint_functor!::S2
    initial_point::Array{T,N}
    new_point::Array{T,N}
    step_direction::Array{T,N}
end

@inline function (so::LineSearchFunctor{S,T})(
        step_size::T) where {S,T<:Real}
    x0, x1, dx = so.initial_point, so.new_point, so.step_direction
    @simd ivdep for i = 1 : length(x0)
        @inbounds x1[i] = x0[i] - step_size * dx[i]
    end
    return so.objective_functor(x1)
end

@inline function (cso::ConstrainedLineSearchFunctor{S1,S2,T,N})(
        step_size::T) where {S1,S2,T<:Real,N}
    x0, x1, dx = cso.initial_point, cso.new_point, cso.step_direction
    @simd ivdep for i = 1 : length(x0)
        @inbounds x1[i] = x0[i] - step_size * dx[i]
    end
    cso.constraint_functor!(x1)
    return cso.objective_functor(x1)
end

########################################################################### BFGS

@enum StepType begin
    NullStep
    GradientDescentStep
    BFGSStep
end

struct BFGSOptimizer{F,G,T}
    num_dims::Int
    objective_function::F
    gradient_function!::G
    iteration_count::Array{Int,0}
    has_converged::Array{Bool,0}
    current_point::Vector{T}
    current_objective_value::Array{T,0}
    current_gradient::Vector{T}
    approximate_inverse_hessian::Matrix{T}
    last_step_size::Array{T,0}
    last_step_type::Array{StepType,0}
    next_step_direction::Vector{T}
    _temp_buffer::Vector{T}
    _delta_gradient::Vector{T}
    _gradient_line_search_functor::LineSearchFunctor{F,T,1}
    _bfgs_line_search_functor::LineSearchFunctor{F,T,1}
end

function BFGSOptimizer(objective_function::F,
                       gradient_function!::G,
                       initial_point::Vector{T},
                       initial_step_size::T) where {F,G,T}
    num_dims = length(initial_point)
    current_point = copy(initial_point)
    current_objective_value = objective_function(current_point)
    current_gradient = Vector{T}(undef, num_dims)
    gradient_function!(current_gradient, current_point)
    next_step_direction = copy(current_gradient)
    _temp_buffer = Vector{T}(undef, num_dims)
    return BFGSOptimizer{F,G,T}(
        num_dims,
        objective_function,
        gradient_function!,
        fill(0),
        fill(false),
        current_point,
        fill(current_objective_value),
        current_gradient,
        identity_matrix!(Matrix{T}(undef, num_dims, num_dims)),
        fill(initial_step_size),
        fill(NullStep),
        next_step_direction,
        _temp_buffer,
        Vector{T}(undef, num_dims),
        LineSearchFunctor{F,T,1}(objective_function,
            current_point, _temp_buffer, current_gradient),
        LineSearchFunctor{F,T,1}(objective_function,
            current_point, _temp_buffer, next_step_direction)
    )
end

function BFGSOptimizer(::Type{T}, opt::BFGSOptimizer{F,G,U}) where {F,G,T,U}
    current_point = T.(opt.current_point)
    current_objective_value = opt.objective_function(current_point)
    current_gradient = Vector{T}(undef, opt.num_dims)
    opt.gradient_function!(current_gradient, current_point)
    next_step_direction = T.(opt.next_step_direction)
    _temp_buffer = Vector{T}(undef, opt.num_dims)
    return BFGSOptimizer{F,G,T}(
        opt.num_dims,
        opt.objective_function,
        opt.gradient_function!,
        fill(opt.iteration_count[]),
        fill(false),
        current_point,
        fill(current_objective_value),
        current_gradient,
        T.(opt.approximate_inverse_hessian),
        fill(T(opt.last_step_size[])),
        fill(opt.last_step_type[]),
        next_step_direction,
        _temp_buffer,
        T.(opt._delta_gradient),
        LineSearchFunctor{F,T,1}(opt.objective_function,
            current_point, _temp_buffer, current_gradient),
        LineSearchFunctor{F,T,1}(opt.objective_function,
            current_point, _temp_buffer, next_step_direction)
    )
end

function BFGSOptimizer(objective_function::F1,
                       gradient_function!::G1,
                       ::Type{T},
                       opt::BFGSOptimizer{F2,G2,U}) where {F1,G1,F2,G2,T,U}
    current_point = T.(opt.current_point)
    current_objective_value = objective_function(current_point)
    current_gradient = Vector{T}(undef, opt.num_dims)
    gradient_function!(current_gradient, current_point)
    next_step_direction = T.(opt.next_step_direction)
    _temp_buffer = Vector{T}(undef, opt.num_dims)
    return BFGSOptimizer{F1,G1,T}(
        opt.num_dims,
        objective_function,
        gradient_function!,
        fill(opt.iteration_count[]),
        fill(false),
        current_point,
        fill(current_objective_value),
        current_gradient,
        T.(opt.approximate_inverse_hessian),
        fill(T(opt.last_step_size[])),
        fill(opt.last_step_type[]),
        next_step_direction,
        _temp_buffer,
        T.(opt._delta_gradient),
        LineSearchFunctor{F1,T,1}(objective_function,
            current_point, _temp_buffer, current_gradient),
        LineSearchFunctor{F1,T,1}(objective_function,
            current_point, _temp_buffer, next_step_direction)
    )
end

function update_inverse_hessian!(
        inv_hess::Matrix{T}, step_size::T, step_direction::Vector{T},
        delta_gradient::Vector{T}, temp_buffer::Vector{T}) where {T}
    overlap = dot(step_direction, delta_gradient)
    step_direction .*= inv(overlap)
    mul!(temp_buffer, inv_hess, delta_gradient)
    delta_norm = step_size * overlap + dot(delta_gradient, temp_buffer)
    n = size(inv_hess, 1)
    @inbounds for j = 1 : n
        sj = step_direction[j]
        tj = temp_buffer[j]
        @simd ivdep for i = 1 : n
            inv_hess[i, j] += (
                delta_norm * (step_direction[i] * sj)
                - (temp_buffer[i] * sj + step_direction[i] * tj))
        end
    end
end

function nanmin(a::T, b::T) where {T}
    if isnan(a)
        return b
    elseif isnan(b)
        return a
    else
        return min(a, b)
    end
end

function step!(opt::BFGSOptimizer{S1,S2,T}) where {S1, S2, T <: Real}

    # Abbreviated names for brevity
    n, x, dg = opt.num_dims, opt.current_point, opt._delta_gradient

    # We actually run two independent line searches on each iteration:
    # one in the quasi-Newton (Hessian)^-1 * (-gradient) direction, and one
    # in the raw (-gradient) direction. The idea is that the BFGS algorithm
    # attempts to incrementally build an approximation to the inverse Hessian.
    # However, if we spend a long time in one region of search space, then
    # quickly move to a different region, our previous Hessian approximation
    # may become unhelpful. By performing this "competitive line search"
    # procedure, we can detect whether this occurs and reset the Hessian
    # when necessary.

    # Use the previous step size as the initial guess for line search
    step_size = opt.last_step_size[]

    # Launch line search in raw (-gradient) direction
    grad_dir = opt.current_gradient
    grad_norm = norm(grad_dir)
    grad_step_size, grad_obj = quadratic_line_search(
        opt._gradient_line_search_functor,
        opt.current_objective_value[], step_size / grad_norm)

    # Launch line search in BFGS (Hessian)^-1 * (-gradient) direction
    bfgs_dir = opt.next_step_direction
    bfgs_norm = norm(bfgs_dir)
    bfgs_step_size, bfgs_obj = quadratic_line_search(
        opt._bfgs_line_search_functor,
        opt.current_objective_value[], step_size / bfgs_norm)

    # We have converged if neither line search reduces the objective function
    opt.has_converged[] = !(
        nanmin(bfgs_obj, grad_obj) < opt.current_objective_value[])
    if opt.has_converged[]; return opt; end
    opt.iteration_count[] += 1

    if bfgs_obj <= grad_obj

        # Accept BFGS step
        opt.current_objective_value[] = bfgs_obj
        opt.last_step_size[] = bfgs_step_size * bfgs_norm
        opt.last_step_type[] = BFGSStep

        # Update point, gradient, and delta_gradient
        @simd ivdep for i = 1 : n
            @inbounds x[i] -= bfgs_step_size * bfgs_dir[i]
        end
        @simd ivdep for i = 1 : n
            @inbounds dg[i] = -grad_dir[i]
        end
        opt.gradient_function!(grad_dir, x)
        @simd ivdep for i = 1 : n
            @inbounds dg[i] += grad_dir[i]
        end

        # Update inverse Hessian approximation using delta_gradient
        update_inverse_hessian!(opt.approximate_inverse_hessian,
            -bfgs_step_size, bfgs_dir, dg, opt._temp_buffer)

        # Compute next step direction using approximate inverse Hessian
        mul!(bfgs_dir, opt.approximate_inverse_hessian, grad_dir)

    else

        # Accept gradient descent step
        opt.current_objective_value[] = grad_obj
        opt.last_step_size[] = grad_step_size * grad_norm
        opt.last_step_type[] = GradientDescentStep

        # Update point and gradient (no need to update delta_gradient,
        # since we'll reset the approximate inverse Hessian)
        @simd ivdep for i = 1 : n
            @inbounds x[i] -= grad_step_size * grad_dir[i]
        end
        opt.gradient_function!(grad_dir, x)

        # Reset approximate inverse Hessian to the identity matrix
        identity_matrix!(opt.approximate_inverse_hessian)

        # Reset next step direction to gradient
        @simd ivdep for i = 1 : n
            @inbounds bfgs_dir[i] = grad_dir[i]
        end

    end

    return opt
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
    step_functor::ConstrainedLineSearchFunctor{S1,S3,T,N}
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
    step_functor = ConstrainedLineSearchFunctor{S1,S3,T,N}(
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
    step_functor::ConstrainedLineSearchFunctor{S1,S3,T,N}
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
    step_functor = ConstrainedLineSearchFunctor{S1,S3,T,N}(
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
        @simd ivdep for j = 1:n; q[j] *= gamma                    ; end
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

end # module DZOptimization
