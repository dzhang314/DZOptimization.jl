module DZOptimization

# export LineSearchFunctor, BFGSOptimizer, step!, find_saturation_threshold,
#     L2RegularizationWrapper, L2GradientWrapper

export LineSearchFunctor, L2RegularizationWrapper, L2GradientWrapper,
    step!, GradientDescentOptimizer, BFGSOptimizer

using LinearAlgebra: mul!


########################################################## UNSAFE LINEAR ALGEBRA


@inline unsafe_sqrt(x::Float32) = Base.sqrt_llvm(x)
@inline unsafe_sqrt(x::Float64) = Base.sqrt_llvm(x)
@inline unsafe_sqrt(x::T) where {T} = sqrt(x)


@inline function norm2(x::AbstractArray{T,N}) where {T,N}
    result = zero(real(T))
    @simd for i = 1 : length(x)
        @inbounds result += abs2(x[i])
    end
    return result
end


@inline function norm(x::AbstractArray{T,N}) where {T,N}
    return unsafe_sqrt(float(norm2(x)))
end


@inline function normalize!(x::AbstractArray{T,N}) where {T,N}
    a = inv(norm(x))
    @simd ivdep for i = 1 : length(x)
        @inbounds x[i] *= a
    end
    return x
end


function normalize_columns!(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    @inbounds for j = 1 : n
        norm_sq = zero(real(T))
        @simd ivdep for i = 1 : m
            norm_sq += abs2(A[i,j])
        end
        inv_norm = inv(unsafe_sqrt(norm_sq))
        @simd ivdep for i = 1 : m
            A[i,j] *= inv_norm
        end
    end
    return A
end


@inline function dot(v::AbstractArray{T,N},
                     w::AbstractArray{T,N}, n::Int) where {T,N}
    result = zero(T)
    @simd ivdep for i = 1 : n
        @inbounds result += conj(v[i]) * w[i]
    end
    result
end


@inline function identity_matrix!(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    for j = 1 : n
        @simd ivdep for i = 1 : m
            @inbounds A[i,j] = ifelse(i == j, one(T), zero(T))
        end
    end
    return A
end


half(::Type{T}) where {T} = one(T) / (one(T) + one(T))


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
        x1 = half(T) * x1
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
            x2 = half(T) * x1
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


############################################################ LINE SEARCH FUNCTOR


struct LineSearchFunctor{F,C,T,N}
    objective_function::F
    constraint_function!::C
    current_point::Array{T,N}
    new_point::Array{T,N}
    step_direction::Array{T,N}
end


@inline function (lsf::LineSearchFunctor{F,C,T,N})(
                  step_size::T) where {F,C,T,N}
    x0, x1, dx = lsf.current_point, lsf.new_point, lsf.step_direction
    @simd ivdep for i = 1 : length(x0)
        @inbounds x1[i] = x0[i] - step_size * dx[i]
    end
    lsf.constraint_function!(x1) && return lsf.objective_function(x1)
    return typemax(T)
end


######################################################## REGULARIZATION WRAPPERS


struct L2RegularizationWrapper{F,T}
    wrapped_function::F
    lambda::T
end


struct L2GradientWrapper{G,T}
    wrapped_gradient!::G
    lambda::T
end


function (wrapper::L2RegularizationWrapper{F,T})(
          x::AbstractArray{T,N}) where {F,T,N}
    return wrapper.wrapped_function(x) + half(T) * wrapper.lambda * norm2(x)
end


function (wrapper::L2GradientWrapper{G,T})(
          g::AbstractArray{T,N}, x::AbstractArray{T,N}) where {G,T,N}
    wrapper.wrapped_gradient!(g, x)
    lambda = wrapper.lambda
    @simd ivdep for i = 1 : length(g)
        @inbounds g[i] += lambda * x[i]
    end
    return g
end


################################################################################


struct GradientDescentOptimizer{F,G,C,T,N}
    objective_function::F
    gradient_function!::G
    constraint_function!::C
    iteration_count::Array{Int,0}
    has_converged::Array{Bool,0}
    current_point::Array{T,N}
    current_objective_value::Array{T,0}
    current_gradient::Array{T,N}
    delta_point::Array{T,N}
    delta_gradient::Array{T,N}
    last_step_size::Array{T,0}
    _scratch_space::Array{T,N}
    _line_search_functor::LineSearchFunctor{F,C,T,N}
end


@inline NULL_CONSTRAINT(_...) = true


function GradientDescentOptimizer(objective_function::F,
                                  gradient_function!::G,
                                  initial_point::Array{T,N},
                                  initial_step_size::T) where {F,G,T,N}
    return GradientDescentOptimizer(
        objective_function, gradient_function!,
        NULL_CONSTRAINT, initial_point, initial_step_size)
end


function GradientDescentOptimizer(objective_function::F,
                                  gradient_function!::G,
                                  constraint_function!::C,
                                  initial_point::Array{T,N},
                                  initial_step_size::T) where {F,G,C,T,N}
    iteration_count = fill(0)
    has_converged = fill(false)
    current_point = copy(initial_point)
    constraint_success = constraint_function!(current_point)
    @assert constraint_success
    initial_objective_value = objective_function(current_point)
    @assert !isnan(initial_objective_value)
    current_objective_value = fill(initial_objective_value)
    current_gradient = similar(initial_point)
    gradient_function!(current_gradient, current_point)
    delta_point = zero(initial_point)
    delta_gradient = zero(initial_point)
    last_step_size = fill(initial_step_size)
    _scratch_space = similar(initial_point)
    _line_search_functor = LineSearchFunctor{F,C,T,N}(
        objective_function, constraint_function!,
        current_point, _scratch_space, current_gradient)
    return GradientDescentOptimizer{F,G,C,T,N}(
        objective_function,
        gradient_function!,
        constraint_function!,
        iteration_count,
        has_converged,
        current_point,
        current_objective_value,
        current_gradient,
        delta_point,
        delta_gradient,
        last_step_size,
        _scratch_space,
        _line_search_functor)
end


function negate!(v::Array{T,N}, w::Array{T,N}, n::Int) where {T,N}
    @simd ivdep for i = 1 : n
        @inbounds v[i] = -w[i]
    end
end


function add!(v::Array{T,N}, w::Array{T,N}, n::Int) where {T,N}
    @simd ivdep for i = 1 : n
        @inbounds v[i] += w[i]
    end
end


function step!(opt::GradientDescentOptimizer{F,G,C,T,N}) where {F,G,C,T,N}
    if !opt.has_converged[]

        point = opt.current_point
        delta_point = opt.delta_point
        gradient = opt.current_gradient
        delta_gradient = opt.delta_gradient

        n = length(point)
        @assert n == length(delta_point)
        @assert n == length(gradient)
        @assert n == length(delta_gradient)

        step_size = opt.last_step_size[]
        gradient_norm = norm(gradient)
        next_step_size, next_obj = quadratic_line_search(
            opt._line_search_functor,
            opt.current_objective_value[],
            step_size / gradient_norm)

        if next_obj < opt.current_objective_value[]

            # Accept gradient descent step.
            opt.current_objective_value[] = next_obj
            opt.last_step_size[] = next_step_size * gradient_norm
            opt.iteration_count[] += 1

            # Update point and gradient.
            negate!(delta_point, point, n)
            negate!(delta_gradient, gradient, n)
            @simd ivdep for i = 1 : n
                @inbounds point[i] -= next_step_size * gradient[i]
            end
            constraint_success = opt.constraint_function!(point)
            @assert constraint_success
            opt.gradient_function!(gradient, point)
            add!(delta_point, point, n)
            add!(delta_gradient, gradient, n)

        else
            opt.has_converged[] = true
        end
    end
    return opt
end


########################################################################### BFGS


@enum StepType begin
    NullStep
    GradientDescentStep
    BFGSStep
end


struct BFGSOptimizer{F,G,C,T,N}
    objective_function::F
    gradient_function!::G
    constraint_function!::C
    iteration_count::Array{Int,0}
    has_converged::Array{Bool,0}
    current_point::Array{T,N}
    current_objective_value::Array{T,0}
    current_gradient::Array{T,N}
    delta_point::Array{T,N}
    delta_gradient::Array{T,N}
    last_step_size::Array{T,0}
    last_step_type::Array{StepType,0}
    approximate_inverse_hessian::Matrix{T}
    next_step_direction::Array{T,N}
    _scratch_space::Array{T,N}
    _gradient_line_search_functor::LineSearchFunctor{F,C,T,N}
    _bfgs_line_search_functor::LineSearchFunctor{F,C,T,N}
end


function BFGSOptimizer(objective_function::F,
                       gradient_function!::G,
                       constraint_function!::C,
                       initial_point::Array{T,N},
                       initial_step_size::T) where {F,G,C,T,N}
    iteration_count = fill(0)
    has_converged = fill(false)
    current_point = copy(initial_point)
    constraint_success = constraint_function!(current_point)
    @assert constraint_success
    initial_objective_value = objective_function(current_point)
    @assert !isnan(initial_objective_value)
    current_objective_value = fill(initial_objective_value)
    current_gradient = similar(initial_point)
    gradient_function!(current_gradient, current_point)
    delta_point = zero(initial_point)
    delta_gradient = zero(initial_point)
    last_step_size = fill(initial_step_size)
    last_step_type = fill(NullStep)
    approximate_inverse_hessian = Matrix{T}(undef,
        length(initial_point), length(initial_point))
    identity_matrix!(approximate_inverse_hessian)
    next_step_direction = copy(current_gradient)
    _scratch_space = similar(initial_point)
    _gradient_line_search_functor = LineSearchFunctor{F,C,T,N}(
        objective_function, constraint_function!,
        current_point, _scratch_space, current_gradient)
    _bfgs_line_search_functor = LineSearchFunctor{F,C,T,N}(
        objective_function, constraint_function!,
        current_point, _scratch_space, next_step_direction)
    return BFGSOptimizer{F,G,C,T,N}(
        objective_function,
        gradient_function!,
        constraint_function!,
        iteration_count,
        has_converged,
        current_point,
        current_objective_value,
        current_gradient,
        delta_point,
        delta_gradient,
        last_step_size,
        last_step_type,
        approximate_inverse_hessian,
        next_step_direction,
        _scratch_space,
        _gradient_line_search_functor,
        _bfgs_line_search_functor)
end


function update_inverse_hessian!(
        inv_hess::Matrix{T}, step_size::T, step_direction::Array{T,N},
        delta_gradient::Array{T,N}, scratch_space::Array{T,N}) where {T,N}

    n = length(step_direction)
    @assert n == length(delta_gradient)
    @assert n == length(scratch_space)
    @assert (n, n) == size(inv_hess)

    overlap = dot(step_direction, delta_gradient, n)
    step_direction .*= inv(overlap)
    mul!(view(scratch_space, :), inv_hess, view(delta_gradient, :))
    delta_norm = step_size * overlap + dot(delta_gradient, scratch_space, n)

    @inbounds for j = 1 : n
        sj = step_direction[j]
        tj = scratch_space[j]
        @simd ivdep for i = 1 : n
            inv_hess[i, j] += (
                delta_norm * (step_direction[i] * sj)
                - (scratch_space[i] * sj + step_direction[i] * tj))
        end
    end

    return inv_hess
end


function step!(opt::BFGSOptimizer{F,G,C,T,N}) where {F,G,C,T,N}
    if !opt.has_converged[]

        point = opt.current_point
        delta_point = opt.delta_point
        gradient = opt.current_gradient
        delta_gradient = opt.delta_gradient
        bfgs_direction = opt.next_step_direction

        n = length(point)
        @assert n == length(delta_point)
        @assert n == length(gradient)
        @assert n == length(delta_gradient)
        @assert n == length(bfgs_direction)

        # We actually run two independent line searches on each iteration:
        # one in the quasi-Newton (Hessian)^-1 * (-gradient) direction, and one
        # in the raw (-gradient) direction. The idea is that the BFGS algorithm
        # attempts to incrementally build an approximation to the inverse
        # Hessian. However, if we spend a long time in one region of search
        # space, then quickly move to a different region, our previous Hessian
        # approximation may become unhelpful. By performing this "competitive
        # line search" procedure, we can detect whether this occurs and reset
        # the Hessian when necessary.

        # Use the previous step size as the initial guess for line search.
        step_size = opt.last_step_size[]

        # Launch line search in raw (-gradient) direction.
        grad_norm = norm(gradient)
        grad_step_size, grad_obj = quadratic_line_search(
            opt._gradient_line_search_functor,
            opt.current_objective_value[],
            step_size / grad_norm)

        # Launch line search in BFGS (Hessian)^-1 * (-gradient) direction.
        bfgs_norm = norm(bfgs_direction)
        bfgs_step_size, bfgs_obj = quadratic_line_search(
            opt._bfgs_line_search_functor,
            opt.current_objective_value[],
            step_size / bfgs_norm)

        if bfgs_obj < opt.current_objective_value[] && !(bfgs_obj > grad_obj)

            # Accept BFGS step.
            opt.current_objective_value[] = bfgs_obj
            opt.last_step_size[] = bfgs_step_size * bfgs_norm
            opt.last_step_type[] = BFGSStep
            opt.iteration_count[] += 1

            # Update point and gradient.
            negate!(delta_point, point, n)
            negate!(delta_gradient, gradient, n)
            @simd ivdep for i = 1 : n
                @inbounds point[i] -= bfgs_step_size * bfgs_direction[i]
            end
            constraint_success = opt.constraint_function!(point)
            @assert constraint_success
            opt.gradient_function!(gradient, point)
            add!(delta_point, point, n)
            add!(delta_gradient, gradient, n)

            # Update inverse Hessian approximation using delta_gradient.
            update_inverse_hessian!(opt.approximate_inverse_hessian,
                                    -bfgs_step_size, bfgs_direction,
                                    delta_gradient, opt._scratch_space)

            # Compute next step direction using approximate inverse Hessian.
            mul!(view(bfgs_direction, :),
                 opt.approximate_inverse_hessian,
                 view(gradient, :))

        elseif grad_obj < opt.current_objective_value[]

            # Accept gradient descent step.
            opt.current_objective_value[] = grad_obj
            opt.last_step_size[] = grad_step_size * grad_norm
            opt.last_step_type[] = GradientDescentStep
            opt.iteration_count[] += 1

            # Update point and gradient.
            negate!(delta_point, point, n)
            negate!(delta_gradient, gradient, n)
            @simd ivdep for i = 1 : n
                @inbounds point[i] -= grad_step_size * gradient[i]
            end
            constraint_success = opt.constraint_function!(point)
            @assert constraint_success
            opt.gradient_function!(gradient, point)
            add!(delta_point, point, n)
            add!(delta_gradient, gradient, n)

            # Reset approximate inverse Hessian to the identity matrix.
            identity_matrix!(opt.approximate_inverse_hessian)

            # Reset next step direction to gradient.
            @simd ivdep for i = 1 : n
                @inbounds bfgs_direction[i] = gradient[i]
            end

        else
            opt.has_converged[] = true
        end
    end
    return opt
end


################################################################## MODEL FITTING


function find_saturation_threshold(data::Vector{Tuple{T,T}}) where {T}

    n = length(data)

    @inbounds sx, sy = data[1]
    sxx = sx * sx
    sxy = sx * sy

    y_mean = zero(T)
    @inbounds y_norm = sy^2
    @simd ivdep for i = 2 : n
        @inbounds y_mean += data[i][2]
        @inbounds y_norm += data[i][2]^2
    end
    y_mean /= (n - 1)

    best_loss = typemax(T)
    best_threshold = zero(T)
    best_slope = zero(T)
    best_mean = zero(T)

    for i = 2 : n-1

        @inbounds xi, yi = data[i]
        sx += xi
        sy += yi
        sxx += xi * xi
        sxy += xi * yi

        y_mean = (y_mean * (n - i + 1) - yi) / (n - i)

        @inbounds lower = data[i][1]
        threshold = lower
        begin
            slope = (i * y_mean - sy) / (i * threshold - sx)
            loss = half(T) * (
                y_norm
                + slope^2 * sxx
                - i * (y_mean - slope * threshold)^2
                - (n - i) * y_mean^2
            ) - slope * sxy
            if loss < best_loss
                best_loss = loss
                best_threshold = threshold
                best_slope = slope
                best_mean = y_mean
            end
        end

        @inbounds upper = data[i+1][1]
        threshold = upper
        begin
            slope = (i * y_mean - sy) / (i * threshold - sx)
            loss = half(T) * (
                y_norm
                + slope^2 * sxx
                - i * (y_mean - slope * threshold)^2
                - (n - i) * y_mean^2
            ) - slope * sxy
            if loss < best_loss
                best_loss = loss
                best_threshold = threshold
                best_slope = slope
                best_mean = y_mean
            end
        end

        a = i * sxx - sx * sx
        b = i * sxy - sx * sy
        c = sx * sxy - sxx * sy
        threshold = (c / b) + (a / b) * y_mean
        if lower <= threshold <= upper
            slope = (i * y_mean - sy) / (i * threshold - sx)
            loss = half(T) * (
                y_norm
                + slope^2 * sxx
                - i * (y_mean - slope * threshold)^2
                - (n - i) * y_mean^2
            ) - slope * sxy
            if loss < best_loss
                best_loss = loss
                best_threshold = threshold
                best_slope = slope
                best_mean = y_mean
            end
        end

    end
    return (best_threshold, best_slope, best_mean, best_loss)
end


################################################################################


# struct ConstrainedLBFGSOptimizer{S1,S2,S3,T<:Real,N}
#     objective_functor::S1
#     gradient_functor!::S2
#     constraint_functor!::S3
#     current_iteration::Vector{Int}
#     current_objective::Vector{T}
#     current_point::Array{T,N}
#     current_gradient::Array{T,N}
#     delta_point::Array{T,N}
#     delta_gradient::Array{T,N}
#     delta_point_history::Matrix{T}
#     delta_gradient_history::Matrix{T}
#     alpha_history::Vector{T}
#     rho_history::Vector{T}
#     history_length::Int
#     step_direction::Array{T,N}
#     scratch_space::Array{T,N}
#     step_functor::ConstrainedLineSearchFunctor{S1,S3,T,N}
# end


# function constrained_lbfgs_optimizer(
#         objective_functor::S1, gradient_functor!::S2, constraint_functor!::S3,
#         initial_point::AbstractArray{T,N}, m::Int) where {S1,S2,S3,T<:Real,N}
#     current_point = copy(initial_point)
#     constraint_functor!(current_point)
#     current_objective = objective_functor(current_point)
#     current_gradient = similar(current_point)
#     gradient_functor!(current_gradient, current_point)
#     delta_point = zero(current_point)
#     delta_gradient = zero(current_gradient)
#     step_direction = similar(current_point)
#     scratch_space = similar(current_point)
#     step_functor = ConstrainedLineSearchFunctor{S1,S3,T,N}(
#         objective_functor, constraint_functor!,
#         current_point, scratch_space, step_direction)
#     return ConstrainedLBFGSOptimizer{S1,S2,S3,T,N}(
#         objective_functor, gradient_functor!, constraint_functor!, Int[0],
#         T[current_objective], current_point, current_gradient,
#         delta_point, delta_gradient,
#         Matrix{T}(undef, length(current_point), m),
#         Matrix{T}(undef, length(current_point), m),
#         Vector{T}(undef, m), Vector{T}(undef, m),
#         m, step_direction, scratch_space, step_functor)
# end


# function step!(opt::ConstrainedLBFGSOptimizer{S1,S2,S3,T,N}
#         ) where {S1,S2,S3,T<:Real,N}
#     @inbounds begin

#         # Define aliases to opt member variables
#         x, g = opt.current_point, opt.current_gradient
#         s, y = opt.delta_point, opt.delta_gradient
#         S, Y = opt.delta_point_history, opt.delta_gradient_history
#         alpha, rho = opt.alpha_history, opt.rho_history
#         m, n, temp = opt.history_length, length(x), opt.scratch_space
#         q, k = opt.step_direction, opt.current_iteration[1] + 1
#         cur_objective = opt.current_objective[1]

#         # Compute step direction
#         @simd ivdep for j = 1:n; q[j] = g[j]                      ; end
#         for i = k-1 : -1 : max(k-m, 1)
#             c = (i - 1) % m + 1
#             d = zero(T)
#             @simd ivdep for j = 1:n; d += S[j,c] * q[j]           ; end
#             a = alpha[c] = rho[c] * d
#             @simd ivdep for j = 1:n; q[j] -= a * Y[j,c]           ; end
#         end
#         gamma = dot(s, y) / norm2(y)
#         if !isfinite(gamma)
#             gamma = sqrt(eps(T)) / max(one(T), norm2(g))
#         end
#         @simd ivdep for j = 1:n; q[j] *= gamma                    ; end
#         for i = max(k-m, 1) : k-1
#             c = (i - 1) % m + 1
#             d = zero(T)
#             @simd ivdep for j = 1:n; d += Y[j,c] * q[j]           ; end
#             beta = alpha[c] - rho[c] * d
#             @simd ivdep for j = 1:n; q[j] += S[j,c] * beta        ; end
#         end

#         # Perform line search
#         step_size, new_objective = quadratic_line_search(
#             opt.step_functor, cur_objective, one(T))

#         # Did we improve? If not, return early
#         if !(new_objective < cur_objective)
#             return false
#         end

#         # If we did improve, accept the step
#         opt.current_iteration[1] = k
#         opt.current_objective[1] = new_objective
#         @simd ivdep for j = 1:n; temp[j] = x[j] - step_size * q[j]; end
#         opt.constraint_functor!(temp)
#         @simd ivdep for j = 1:n; s[j] = temp[j] - x[j]            ; end
#         @simd ivdep for j = 1:n; x[j] = temp[j]                   ; end
#         opt.gradient_functor!(temp, x)
#         @simd ivdep for j = 1:n; y[j] = temp[j] - g[j]            ; end
#         @simd ivdep for j = 1:n; g[j] = temp[j]                   ; end
#         c = (k - 1) % m + 1
#         @simd ivdep for j = 1:n; S[j,c] = s[j]                    ; end
#         @simd ivdep for j = 1:n; Y[j,c] = y[j]                    ; end
#         rho[c] = inv(dot(s, y))
#     end
#     return true
# end


################################################################################


include("./ExampleFunctions.jl")

end # module DZOptimization
