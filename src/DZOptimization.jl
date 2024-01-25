module DZOptimization


using MultiFloats: MultiFloat, MultiFloatVec, rsqrt


################################################################################


@inline _iota(::Val{M}) where {M} = Vec{M,Int}(ntuple(i -> i - 1, Val{M}()))


function norm2(x::AbstractArray{T,D}) where {T,D}
    result = zero(real(T))
    @simd for i in eachindex(x)
        result += abs2(x[i])
    end
    return result
end


function norm2_mfv(x::Array{MultiFloat{T,N},D}, ::Val{M}) where {M,T,N,D}
    n = length(x)
    ptr = pointer(x)
    iota = _iota(Val{M}()) - 1
    i = 1
    result_vector = zero(MultiFloatVec{M,T,N})
    while i + M <= n + 1
        result_vector += abs2(mfvgather(ptr, iota + i))
        i += M
    end
    result_scalar = zero(MultiFloat{T,N})
    @inbounds while i <= n
        result_scalar += abs2(x[i])
        i += 1
    end
    return result_scalar + sum(result_vector)
end


@inline norm2(x::Array{MultiFloat{T,N},D}) where {T,N,D} =
    norm2_mfv(x, Val{8}())


@inline inv_norm(x::AbstractArray{T,D}) where {T,D} = rsqrt(norm2(x))


function scale!(x::AbstractArray{T,D}, alpha::T) where {T,D}
    @simd ivdep for i in eachindex(x)
        x[i] *= alpha
    end
    return x
end


################################################################################


@inline NULL_CONSTRAINT(_...) = true


################################################################################


struct GradientDescentOptimizer{C,F,G,L,T,N}

    constraint_function!::C
    current_point::Array{T,N}
    delta_point::Array{T,N}

    objective_function::F
    current_objective_value::Array{T,0}
    delta_objective_value::Array{T,0}

    gradient_function!::G
    current_gradient::Array{T,N}
    delta_gradient::Array{T,N}

    line_search_function!::L
    next_step_direction::Array{T,N}
    last_step_size::Array{T,0}

    iteration_count::Array{Int,0}
    has_converged::Array{Bool,0}

end


function GradientDescentOptimizer(
    constraint_function!::C,
    objective_function::F,
    gradient_function!::G,
    line_search_function!::L,
    initial_point::AbstractArray{T,N},
    initial_step_size::T,
) where {C,F,G,L,T,N}

    current_point = collect(initial_point)
    @assert constraint_function!(current_point)
    delta_point = zero(current_point)

    initial_objective_value = objective_function(current_point)
    @assert isfinite(initial_objective_value)
    current_objective_value = fill(initial_objective_value)
    delta_objective_value = zeros(T)

    current_gradient = similar(current_point)
    gradient_function!(current_gradient, current_point)
    delta_gradient = zero(current_point)

    last_step_size = fill(initial_step_size)
    next_step_direction = scale!(copy(current_gradient),
        initial_step_size * inv_norm(current_gradient))

    iteration_count = fill(0)
    has_converged = fill(false)

    return GradientDescentOptimizer{C,F,G,L,T,N}(
        constraint_function!, current_point, delta_point,
        objective_function, current_objective_value, delta_objective_value,
        gradient_function!, current_gradient, delta_gradient,
        line_search_function!, next_step_direction, last_step_size,
        iteration_count, has_converged)
end


function step!(opt::GradientDescentOptimizer{F,G,C,T,N}) where {F,G,C,T,N}

    n = length(opt.current_point)
    @assert n == length(opt.delta_point)
    @assert n == length(opt.current_gradient)
    @assert n == length(opt.delta_gradient)
    @assert n == length(opt.next_step_direction)

    if !opt.has_converged[]

        #=
        step_size = opt.last_step_size[]
        gradient_norm = norm(opt.current_gradient)
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
            negate!(opt.delta_point, opt.current_point, n)
            negate!(opt.delta_gradient, opt.current_gradient, n)
            add!(opt.current_point, -next_step_size, opt.current_gradient, n)
            constraint_success = opt.constraint_function!(opt.current_point)
            @assert constraint_success
            opt.gradient_function!(opt.current_gradient, opt.current_point)
            add!(opt.delta_point, opt.current_point, n)
            add!(opt.delta_gradient, opt.current_gradient, n)

        else
            opt.has_converged[] = true
        end
        =#

    end
    return opt
end


################################################################################


struct LBFGSOptimizer{C,F,G,L,T,N}

    constraint_function!::C
    current_point::Array{T,N}
    delta_point::Array{T,N}

    objective_function::F
    current_objective_value::Array{T,0}
    delta_objective_value::Array{T,0}

    gradient_function!::G
    current_gradient::Array{T,N}
    delta_gradient::Array{T,N}

    line_search_function!::L
    next_step_direction::Array{T,N}
    last_step_size::Array{T,0}

    iteration_count::Array{Int,0}
    has_converged::Array{Bool,0}

    _alpha_history::Vector{T}
    _rho_history::Vector{T}
    _delta_point_history::Matrix{T}
    _delta_gradient_history::Matrix{T}

end


function LBFGSOptimizer(
    constraint_function!::C,
    objective_function::F,
    gradient_function!::G,
    line_search_function!::L,
    initial_point::AbstractArray{T,N},
    initial_step_size::T,
    history_length::Int,
) where {C,F,G,L,T,N}

    current_point = collect(initial_point)
    @assert constraint_function!(current_point)
    delta_point = zero(current_point)

    initial_objective_value = objective_function(current_point)
    @assert isfinite(initial_objective_value)
    current_objective_value = fill(initial_objective_value)
    delta_objective_value = zeros(T)

    current_gradient = similar(current_point)
    gradient_function!(current_gradient, current_point)
    delta_gradient = zero(current_point)

    last_step_size = fill(initial_step_size)
    next_step_direction = scale!(copy(current_gradient),
        initial_step_size * inv_norm(current_gradient))

    iteration_count = fill(0)
    has_converged = fill(false)

    @assert history_length > 0
    _alpha_history = zeros(T, history_length)
    _rho_history = zeros(T, history_length)
    _delta_point_history = zeros(T, length(current_point), history_length)
    _delta_gradient_history = zeros(T, length(current_point), history_length)

    return LBFGSOptimizer{C,F,G,L,T,N}(
        constraint_function!, current_point, delta_point,
        objective_function, current_objective_value, delta_objective_value,
        gradient_function!, current_gradient, delta_gradient,
        line_search_function!, next_step_direction, last_step_size,
        iteration_count, has_converged,
        _alpha_history, _rho_history,
        _delta_point_history, _delta_gradient_history)
end


function step!(opt::LBFGSOptimizer{C,F,G,L,T,N}) where {C,F,G,L,T,N}

    n = length(opt.current_point)
    @assert n == length(opt.delta_point)
    @assert n == length(opt.current_gradient)
    @assert n == length(opt.delta_gradient)
    @assert n == length(opt.next_step_direction)

    m = length(opt._alpha_history)
    @assert m == length(opt._rho_history)
    @assert (n, m) == size(opt._delta_point_history)
    @assert (n, m) == size(opt._delta_gradient_history)

    if !opt.has_converged[]



    end
    return opt
end


#=

        next_step_size, next_obj = quadratic_line_search(
            opt._line_search_functor,
            opt.current_objective_value[],
            one(T))

        if next_obj < opt.current_objective_value[]

            # Accept L-BFGS step.
            opt.current_objective_value[] = next_obj
            opt.last_step_size[] = next_step_size * norm(next_step_direction)
            opt.iteration_count[] += 1

            # Update point and gradient.
            negate!(delta_point, point, n)
            negate!(delta_gradient, gradient, n)
            add!(point, -next_step_size, next_step_direction, n)
            constraint_success = opt.constraint_function!(point)
            @assert constraint_success
            opt.gradient_function!(gradient, point)
            add!(delta_point, point, n)
            add!(delta_gradient, gradient, n)

            # Store delta_point and delta_gradient in history.
            c = Base.srem_int(opt.iteration_count[] - 1, m) + 1 # cyclic index
            @simd ivdep for i = 1:n
                @inbounds delta_point_history[i, c] = delta_point[i]
            end
            @simd ivdep for i = 1:n
                @inbounds delta_gradient_history[i, c] = delta_gradient[i]
            end
            delta_overlap = dot(delta_point, delta_gradient, n)
            @inbounds rho[c] = inv(delta_overlap)

            # Compute next step direction, starting from current gradient.
            @simd ivdep for i = 1:n
                @inbounds next_step_direction[i] = gradient[i]
            end

            # Apply forward L-BFGS correction.
            history_count = max(opt.iteration_count[] - m + 1, 1)
            for iter = opt.iteration_count[]:-1:history_count
                c = Base.srem_int(iter - 1, m) + 1
                @inbounds overlap = rho[c] * dot(
                    next_step_direction, delta_point_history, n, c)
                @inbounds alpha[c] = overlap
                add!(next_step_direction,
                    overlap, delta_gradient_history, n, c)
            end

            # Compute natural step size.
            gamma = delta_overlap / norm2(delta_gradient)
            if !isfinite(gamma)
                gamma = sqrt(eps(T)) / max(one(T), norm2(gradient))
            end
            scalar_mul!(next_step_direction, gamma)

            # Apply backward L-BFGS correction.
            for iter = history_count:opt.iteration_count[]
                c = Base.srem_int(iter - 1, m) + 1
                @inbounds overlap = alpha[c] - rho[c] * dot(
                    next_step_direction, delta_gradient_history, n, c)
                add!(next_step_direction,
                    overlap, delta_point_history, n, c)
            end

        else
            opt.has_converged[] = true
        end

export LineSearchFunctor, L2RegularizationWrapper, L2GradientWrapper,
    step!, GradientDescentOptimizer, BFGSOptimizer, LBFGSOptimizer

using LinearAlgebra: mul!

########################################################## UNSAFE LINEAR ALGEBRA

@inline half(::Type{T}) where {T} = one(T) / (one(T) + one(T))

@inline function normalize!(x::AbstractArray{T,N}) where {T,N}
    a = inv(norm(x))
    @simd ivdep for i = 1 : length(x)
        @inbounds x[i] *= a
    end
    return x
end

@inline function dot(v::AbstractArray{T,N},
                     w::AbstractArray{T,N}, n::Int) where {T,N}
    result = zero(T)
    @simd ivdep for i = 1 : n
        @inbounds result += conj(v[i]) * w[i]
    end
    return result
end

@inline function dot(v::AbstractArray{T,N},
                     w::AbstractMatrix{T}, n::Int, j::Int) where {T,N}
    result = zero(T)
    @simd ivdep for i = 1 : n
        @inbounds result += conj(v[i]) * w[i,j]
    end
    return result
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

@inline function negate!(v::AbstractArray{T,N},
                         w::AbstractArray{T,N}, n::Int) where {T,N}
    @simd ivdep for i = 1 : n
        @inbounds v[i] = -w[i]
    end
    return v
end

@inline function add!(v::AbstractArray{T,N},
                      w::AbstractArray{T,N}, n::Int) where {T,N}
    @simd ivdep for i = 1 : n
        @inbounds v[i] += w[i]
    end
    return v
end

@inline function add!(v::AbstractArray{T,N}, c::T,
                      w::AbstractArray{T,N}, n::Int) where {T,N}
    @simd ivdep for i = 1 : n
        @inbounds v[i] += c * w[i]
    end
    return v
end

@inline function add!(v::AbstractArray{T,N}, c::T,
                      w::AbstractMatrix{T}, n::Int, j::Int) where {T,N}
    @simd ivdep for i = 1 : n
        @inbounds v[i] += c * w[i,j]
    end
    return v
end

@inline function scalar_mul!(v::AbstractArray{T,N}, c::T) where {T,N}
    @simd ivdep for i = 1 : length(v)
        @inbounds v[i] *= c
    end
    return v
end

@inline linear_view(x::Array{T,N}) where {T,N} =
    reshape(view(x, ntuple(_ -> Colon(), N)...), length(x))

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

function quadratic_line_search(f::F, f0::T, x1::T,
                               max_increases::Int=10,
                               max_decreases::Int=100) where {F,T}
    # TODO: In principle, we could make this work for f0 == +Inf.
    if !isfinite(f0) || !isfinite(x1)
        return zero(T), f0
    end
    f1 = f(x1)
    while isnan(f1)
        x1 = half(T) * x1
        if iszero(x1)
            return zero(T), f0
        end
        f1 = f(x1)
    end
    if f1 < f0
        num_increases = 0
        while true
            x2 = x1 + x1
            f2 = f(x2)
            num_increases += 1
            if (f2 >= f1) || isnan(f2) || (num_increases > max_increases)
                x3 = x1 * _qls_minimum_high(f0, f1, f2)
                return _qls_best(f0, x1, f1, x2, f2, x3, f(x3))
            else
                x1, f1 = x2, f2
            end
        end
    else
        num_decreases = 0
        while true
            x2 = half(T) * x1
            f2 = f(x2)
            num_decreases += 1
            if isnan(f2)
                return (zero(T), f0)
            end
            if (f2 <= f0) || (num_decreases > max_decreases)
                x3 = x1 * _qls_minimum_low(f0, f1, f2)
                return _qls_best(f0, x2, f2, x1, f1, x3, f(x3))
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
                       initial_point::Array{T,N},
                       initial_step_size::T) where {F,G,T,N}
    return BFGSOptimizer(
        objective_function, gradient_function!,
        NULL_CONSTRAINT, initial_point, initial_step_size)
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

function BFGSOptimizer(::Type{T},
                       opt::BFGSOptimizer{F,G,C,U,N}) where {F,G,C,U,T,N}
    return BFGSOptimizer(T, opt.objective_function, opt.gradient_function!,
                         opt.constraint_function!, opt)
end


function BFGSOptimizer(::Type{T},
                       objective_function::F,
                       gradient_function!::G,
                       constraint_function!::C,
                       opt::BFGSOptimizer{F2,G2,C2,T2,N}
                       ) where {F,G,C,T,F2,G2,C2,T2,N}
    current_point = T.(opt.current_point)
    constraint_success = constraint_function!(current_point)
    @assert constraint_success
    initial_objective_value = objective_function(current_point)
    @assert !isnan(initial_objective_value)
    current_gradient = similar(current_point)
    gradient_function!(current_gradient, current_point)
    approximate_inverse_hessian = T.(opt.approximate_inverse_hessian)
    next_step_direction = similar(current_point)
    mul!(linear_view(next_step_direction),
         approximate_inverse_hessian,
         linear_view(current_gradient))
    _scratch_space = similar(current_point)
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
        fill(opt.iteration_count[]),
        fill(false),
        current_point,
        fill(initial_objective_value),
        current_gradient,
        T.(opt.delta_point),
        T.(opt.delta_gradient),
        fill(opt.last_step_size[]),
        fill(opt.last_step_type[]),
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
    scalar_mul!(step_direction, inv(overlap))
    mul!(linear_view(scratch_space), inv_hess, linear_view(delta_gradient))
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
            add!(point, -bfgs_step_size, bfgs_direction, n)
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
            mul!(linear_view(bfgs_direction),
                 opt.approximate_inverse_hessian,
                 linear_view(gradient))

        elseif grad_obj < opt.current_objective_value[]

            # Accept gradient descent step.
            opt.current_objective_value[] = grad_obj
            opt.last_step_size[] = grad_step_size * grad_norm
            opt.last_step_type[] = GradientDescentStep
            opt.iteration_count[] += 1

            # Update point and gradient.
            negate!(delta_point, point, n)
            negate!(delta_gradient, gradient, n)
            add!(point, -grad_step_size, gradient, n)
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

############################################################## OPTIMIZER TESTING

function run_and_test!(opt)

    history = [deepcopy(opt)]
    while !opt.has_converged[]
        step!(opt)
        push!(history, deepcopy(opt))
    end

    # None of the optimizers should have converged, except the last one.
    for i = 1 : length(history) - 1
        @assert !history[i].has_converged[]
    end
    @assert history[end].has_converged[]

    # Verify consistency of opt.iteration_count.
    for i = 1 : length(history) - 1
        @assert history[i].iteration_count[] == i - 1
    end
    history[end-1].iteration_count[] == history[end].iteration_count[]

    # Verify consistency of opt.current_objective_value.
    for opt in history
        @assert opt.objective_function(opt.current_point) ==
                opt.current_objective_value[]
    end

    # Verify consistency of opt.current_gradient.
    for opt in history
        grad = similar(opt.current_gradient)
        opt.gradient_function!(grad, opt.current_point)
        if grad != opt.current_gradient
            println("ERROR: ", norm(grad - opt.current_gradient))
        end
        @assert grad == opt.current_gradient
    end

    # Verify consistency of opt.delta_point.
    for i = 1 : length(history) - 2
        delta_point = history[i+1].current_point - history[i].current_point
        @assert delta_point == history[i+1].delta_point
    end
    @assert history[end-1].current_point == history[end].current_point

    # Verify consistency of opt.delta_gradient.
    for i = 1 : length(history) - 2
        delta_grad = history[i+1].current_gradient - history[i].current_gradient
        @assert delta_grad == history[i+1].delta_gradient
    end
    @assert history[end-1].current_gradient == history[end].current_gradient

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

=#

end # module DZOptimization
