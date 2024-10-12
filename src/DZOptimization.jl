module DZOptimization


include("SortingNetworks.jl")


#=

include("ExampleFunctions.jl")
include("PCG.jl")
include("Kernels.jl")
using .Kernels: dot, norm2, inv_norm, negate!, scale!, delta!, axpy!


######################################################### OPTIMIZATION UTILITIES


@inline NULL_CONSTRAINT(_...) = true


struct LineSearchEvaluator{C,F,T,N}

    constraint_function!::C
    initial_point::Array{T,N}
    new_point::Array{T,N}
    reference_point::Array{T,N}

    objective_function::F
    step_direction::Array{T,N}

end


function (lse::LineSearchEvaluator{C,F,T,N})(step_size::T) where {C,F,T,N}

    # Validate array sizes.
    n = length(lse.initial_point)
    @assert n == length(lse.new_point)
    @assert n == length(lse.step_direction)

    # Compute new point from initial point, step direction, and step size.
    axpy!(lse.new_point, step_size, lse.step_direction, lse.initial_point, n)

    # Apply constraint function to project new point into feasible region.
    is_feasible = lse.constraint_function!(lse.new_point)

    # If projection fails, we return an objective value of positive infinity
    # to indicate that the new point is infeasible.
    if !is_feasible
        return typemax(T)
    end

    return lse.objective_function(lse.new_point)

end


function find_three_point_bracket(
    lse::LineSearchEvaluator{C,F,T,N}, f0::T, max_increases::Int
) where {C,F,T,N}

    # Validate array sizes.
    n = length(lse.initial_point)
    @assert n == length(lse.new_point)
    @assert n == length(lse.reference_point)
    @assert n == length(lse.step_direction)

    # Construct constants.
    _zero = zero(T)
    _one = one(T)

    # Exit early if initial objective value is non-finite.
    if !isfinite(f0)
        return (_zero, f0, _zero, f0)
    end

    # In some cases, the step vector can be so small (on the order of rounding
    # error) that adding it to the initial point has no effect. We detect this
    # condition by comparing the coordinates of the initial and new points.
    step_is_zero = true
    point_changed = false
    @inbounds for i = 1:n
        step = lse.step_direction[i]
        step_is_zero &= iszero(step)
        initial = lse.initial_point[i]
        new = initial + step
        point_changed |= (initial != new)
        lse.new_point[i] = new
    end

    # Exit early if step vector is zero.
    if step_is_zero
        return (_zero, f0, _zero, f0)
    end

    # If the step vector is nonzero but too small to have an effect,
    # then we repeatedly double the step size until it is large enough.
    step_size = _one
    step_is_small = false
    while !point_changed
        step_size += step_size
        step_is_small = true
        point_changed = false
        @inbounds for i = 1:n
            initial = lse.initial_point[i]
            new = initial + step_size * lse.step_direction[i]
            point_changed |= (initial != new)
            lse.new_point[i] = new
        end
    end

    # Apply constraint function to project new point into feasible region.
    is_feasible = lse.constraint_function!(lse.new_point)

    # Exit early if initial point is on the boundary of the feasible region.
    if step_is_small

        # If the new point is infeasible even after taking the smallest
        # possible step, then the initial point must have been on the boundary.
        if !is_feasible
            return (_zero, f0, _zero, f0)
        end

        # If the new point is identical to the initial point after
        # applying constraints, then it must have been on the boundary.
        # NOTE: We cannot use `point_changed` here because `lse.new_point`
        # may have been modified by applying the constraint function.
        if lse.initial_point == lse.new_point
            return (_zero, f0, _zero, f0)
        end

    end

    # Evaluate objective function at new point.
    f1 = is_feasible ? lse.objective_function(lse.new_point) : typemax(T)

    # If the new point is better than the initial point, then we repeatedly
    # take larger steps until we find a worse or infeasible point.
    if f1 <= f0

        # In some cases, repeatedly taking larger steps may cause us to get
        # stuck at the boundary of the feasible region. To detect this
        # condition, we keep track of the last two feasible points and
        # terminate the search if they coincide.
        copy!(lse.reference_point, lse.new_point)

        # A user may optionally specify a maximum number of times to increase
        # (double) the step size to prevent the search from straying too far.
        # If `max_increases` is zero or negative, then no limit is imposed.
        num_increases = 0

        while true
            double_step_size = step_size + step_size
            num_increases += 1
            f2 = lse(double_step_size)
            if (((max_increases > 0) && (num_increases >= max_increases)) ||
                (!isfinite(f2)) ||
                (f2 > f1) ||
                (lse.new_point == lse.reference_point))
                return (step_size, f1, double_step_size, f2)
            end
            step_size = double_step_size
            f1 = f2
            copy!(lse.reference_point, lse.new_point)
        end
    else

        # If the new point is worse than the initial point, then we repeatedly
        # take smaller steps until we find a better feasible point.
        _half = inv(_one + _one)
        while true
            half_step_size = _half * step_size
            f2 = lse(half_step_size)
            if f2 <= f0
                return (half_step_size, f2, step_size, f1)
            end
            step_size = half_step_size
            f1 = f2
        end
    end
end


########################################################## QUADRATIC LINE SEARCH


export QuadraticLineSearch


struct QuadraticLineSearch
    max_increases::Int
end


function QuadraticLineSearch()
    return QuadraticLineSearch(0)
end


function (qls::QuadraticLineSearch)(
    lse::LineSearchEvaluator{C,F,T,N}, f0::T, _::Array{T,N}
) where {C,F,T,N}
    _zero = zero(T)
    (x1, f1, x2, f2) = find_three_point_bracket(lse, f0, qls.max_increases)
    xb, fb = _zero, f0
    if f1 < fb
        xb, fb = x1, f1
    end
    if f2 < fb
        xb, fb = x2, f2
    end
    delta_1 = f0 - f1
    delta_2 = f2 - f1
    sum_deltas = delta_1 + delta_2
    if (delta_1 >= _zero) && (delta_2 >= _zero) && (sum_deltas > _zero)
        twice_delta_1 = delta_1 + delta_1
        delta_ratio = (twice_delta_1 + sum_deltas) / (sum_deltas + sum_deltas)
        xq = delta_ratio * x1
        fq = lse(xq)
        if fq < fb
            xb, fb = xq, fq
        end
    end
    return (xb, fb)
end


############################################################## L2 REGULARIZATION


export L2RegularizationWrapper, L2GradientWrapper


struct L2RegularizationWrapper{F,T}
    objective_function::F
    lambda::T
end


@inline (wrapper::L2RegularizationWrapper{F,T})(x::Array{T,D}) where {F,T,D} =
    wrapper.objective_function(x) + wrapper.lambda * norm2(x)


struct L2GradientWrapper{G,T}
    gradient_function!::G
    lambda::T
end


@inline function (wrapper::L2GradientWrapper{G,T})(
    g::Array{T,D}, x::Array{T,D}
) where {G,T,D}
    n = length(g)
    @assert n == length(x)
    wrapper.gradient_function!(g, x)
    axpy!(g, wrapper.lambda + wrapper.lambda, x, n)
    return g
end


################################################################ BOX CONSTRAINTS


export UniformBoxConstraint, UniformBoxGradientWrapper


struct UniformBoxConstraint{T}
    lower_bound::T
    upper_bound::T
end


@inline function (constraint::UniformBoxConstraint{T})(
    x::Array{T,D}
) where {T,D}
    n = length(x)
    @simd ivdep for i = 1:n
        x[i] = clamp(x[i], constraint.lower_bound, constraint.upper_bound)
    end
    return true
end


struct UniformBoxGradientWrapper{G,T}
    gradient_function!::G
    lower_bound::T
    upper_bound::T
end


@inline function (wrapper::UniformBoxGradientWrapper{G,T})(
    g::Array{T,N}, x::Array{T,N}
) where {G,T,N}
    n = length(g)
    @assert n == length(x)
    wrapper.gradient_function!(g, x)
    _zero = zero(T)
    @simd ivdep for i = 1:n
        if (((x[i] <= wrapper.lower_bound) && (g[i] >= _zero)) ||
            ((x[i] >= wrapper.upper_bound) && (g[i] <= _zero)))
            g[i] = _zero
        end
    end
    return g
end


############################################################### GRADIENT DESCENT


export GradientDescentOptimizer, step!


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
    last_step_length::Array{T,0}
    line_search_evaluator::LineSearchEvaluator{C,F,T,N}

    iteration_count::Array{Int,0}
    has_terminated::Array{Bool,0}

end


function GradientDescentOptimizer(
    constraint_function!::C,
    objective_function::F,
    gradient_function!::G,
    line_search_function!::L,
    initial_point::AbstractArray{T,N},
    initial_step_length::T,
) where {C,F,G,L,T,N}

    current_point = collect(initial_point)
    @assert constraint_function!(current_point)
    delta_point = zero(current_point)

    initial_objective_value = objective_function(current_point)
    current_objective_value = fill(initial_objective_value)
    delta_objective_value = zeros(T)

    current_gradient = similar(current_point)
    gradient_function!(current_gradient, current_point)
    delta_gradient = zero(current_point)

    last_step_length = fill(zero(T))
    inv_gradient_norm = inv_norm(current_gradient)
    next_step_direction = zero(current_point)
    if isfinite(inv_gradient_norm)
        copy!(next_step_direction, current_gradient)
        scale!(next_step_direction, -initial_step_length * inv_gradient_norm)
    end
    line_search_evaluator = LineSearchEvaluator{C,F,T,N}(
        constraint_function!, current_point,
        similar(current_point), similar(current_point),
        objective_function, next_step_direction)

    iteration_count = fill(0)
    has_terminated = fill(
        (!isfinite(initial_objective_value)) ||
        (!isfinite(inv_gradient_norm)))

    return GradientDescentOptimizer{C,F,G,L,T,N}(
        constraint_function!, current_point, delta_point,
        objective_function, current_objective_value, delta_objective_value,
        gradient_function!, current_gradient, delta_gradient,
        line_search_function!, next_step_direction, last_step_length,
        line_search_evaluator, iteration_count, has_terminated)
end


GradientDescentOptimizer(
    objective_function::F,
    gradient_function!::G,
    line_search_function!::L,
    initial_point::AbstractArray{T,N},
    initial_step_length::T,
) where {F,G,L,T,N} = GradientDescentOptimizer(
    NULL_CONSTRAINT,
    objective_function,
    gradient_function!,
    line_search_function!,
    initial_point,
    initial_step_length,
)


function step!(opt::GradientDescentOptimizer{C,F,G,L,T,N}) where {C,F,G,L,T,N}

    # Validate array sizes.
    n = length(opt.current_point)
    @assert n == length(opt.delta_point)
    @assert n == length(opt.current_gradient)
    @assert n == length(opt.delta_gradient)
    @assert n == length(opt.next_step_direction)

    if !opt.has_terminated[]

        # Perform line search.
        step_size, objective_value = opt.line_search_function!(
            opt.line_search_evaluator,
            opt.current_objective_value[], opt.current_gradient)

        # If line search yielded no improvement, terminate.
        if (iszero(step_size) ||
            !(objective_value < opt.current_objective_value[]))
            opt.has_terminated[] = true
            return opt
        end
        opt.iteration_count[] += 1

        # Update current point and apply constraints.
        copy!(opt.delta_point, opt.current_point)
        axpy!(opt.current_point, step_size, opt.next_step_direction, n)
        @assert opt.constraint_function!(opt.current_point)

        # Compute delta point and step length.
        delta!(opt.delta_point, opt.current_point, n)
        step_length = sqrt(norm2(opt.delta_point))
        opt.last_step_length[] = step_length

        # Update objective value and delta.
        opt.delta_objective_value[] =
            objective_value - opt.current_objective_value[]
        opt.current_objective_value[] = objective_value

        # Update gradient and delta.
        copy!(opt.delta_gradient, opt.current_gradient)
        opt.gradient_function!(opt.current_gradient, opt.current_point)
        delta!(opt.delta_gradient, opt.current_gradient, n)

        # If gradient is zero or non-finite, terminate.
        inv_gradient_norm = inv_norm(opt.current_gradient)
        if !isfinite(inv_gradient_norm)
            opt.has_terminated[] = true
            return opt
        end

        # Compute next step direction.
        scale!(opt.next_step_direction,
            -step_length * inv_gradient_norm, opt.current_gradient, n)
    end
    return opt
end


######################################################################### L-BFGS


export LBFGSOptimizer, step!


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
    last_step_length::Array{T,0}
    line_search_evaluator::LineSearchEvaluator{C,F,T,N}

    iteration_count::Array{Int,0}
    has_terminated::Array{Bool,0}

    _alpha::Vector{T}
    _rho::Vector{T}
    _delta_point_history::Matrix{T}
    _delta_gradient_history::Matrix{T}
    _history_count::Array{Int,0}

end


function LBFGSOptimizer(
    constraint_function!::C,
    objective_function::F,
    gradient_function!::G,
    line_search_function!::L,
    initial_point::AbstractArray{T,N},
    initial_step_length::T,
    history_length::Int,
) where {C,F,G,L,T,N}

    current_point = collect(initial_point)
    @assert constraint_function!(current_point)
    delta_point = zero(current_point)

    initial_objective_value = objective_function(current_point)
    current_objective_value = fill(initial_objective_value)
    delta_objective_value = zeros(T)

    current_gradient = similar(current_point)
    gradient_function!(current_gradient, current_point)
    delta_gradient = zero(current_point)

    last_step_length = fill(zero(T))
    inv_gradient_norm = inv_norm(current_gradient)
    next_step_direction = zero(current_point)
    if isfinite(inv_gradient_norm)
        copy!(next_step_direction, current_gradient)
        scale!(next_step_direction, -initial_step_length * inv_gradient_norm)
    end
    line_search_evaluator = LineSearchEvaluator{C,F,T,N}(
        constraint_function!, current_point,
        similar(current_point), similar(current_point),
        objective_function, next_step_direction)

    iteration_count = fill(0)
    has_terminated = fill(
        (!isfinite(initial_objective_value)) ||
        (!isfinite(inv_gradient_norm)))

    @assert history_length > 0
    _alpha = Vector{T}(undef, history_length)
    _rho = Vector{T}(undef, history_length)
    _delta_point_history =
        Matrix{T}(undef, length(current_point), history_length)
    _delta_gradient_history =
        Matrix{T}(undef, length(current_point), history_length)
    _history_count = fill(0)

    return LBFGSOptimizer{C,F,G,L,T,N}(
        constraint_function!, current_point, delta_point,
        objective_function, current_objective_value, delta_objective_value,
        gradient_function!, current_gradient, delta_gradient,
        line_search_function!, next_step_direction, last_step_length,
        line_search_evaluator, iteration_count, has_terminated, _alpha, _rho,
        _delta_point_history, _delta_gradient_history, _history_count)
end


LBFGSOptimizer(
    objective_function::F,
    gradient_function!::G,
    line_search_function!::L,
    initial_point::AbstractArray{T,N},
    initial_step_length::T,
    history_length::Int,
) where {F,G,L,T,N} = LBFGSOptimizer(
    NULL_CONSTRAINT,
    objective_function,
    gradient_function!,
    line_search_function!,
    initial_point,
    initial_step_length,
    history_length,
)


function step!(opt::LBFGSOptimizer{C,F,G,L,T,N}) where {C,F,G,L,T,N}

    # Validate array sizes.
    n = length(opt.current_point)
    @assert n == length(opt.delta_point)
    @assert n == length(opt.current_gradient)
    @assert n == length(opt.delta_gradient)
    @assert n == length(opt.next_step_direction)
    m = length(opt._alpha)
    @assert m == length(opt._rho)
    @assert (n, m) == size(opt._delta_point_history)
    @assert (n, m) == size(opt._delta_gradient_history)

    if !opt.has_terminated[]

        # Construct constants.
        _zero = zero(T)

        # Perform line search.
        step_size, objective_value = opt.line_search_function!(
            opt.line_search_evaluator,
            opt.current_objective_value[], opt.current_gradient)

        # If line search yielded no improvement, ...
        if (iszero(step_size) ||
            !(objective_value < opt.current_objective_value[]))

            # ... try again using uncorrected gradient as step direction.
            copy!(opt.next_step_direction, opt.current_gradient)
            scale!(opt.next_step_direction,
                -opt.last_step_length[] * inv_norm(opt.current_gradient))
            step_size, objective_value = opt.line_search_function!(
                opt.line_search_evaluator,
                opt.current_objective_value[], opt.current_gradient)

            # If line search still yields no improvement, terminate.
            if (iszero(step_size) ||
                !(objective_value < opt.current_objective_value[]))
                opt.has_terminated[] = true
                return opt
            end

            # If gradient line search yields an improvement,
            # we reset the Hessian approximation.
            opt._history_count[] = 0
        end
        opt.iteration_count[] += 1

        # Update current point and apply constraints.
        copy!(opt.delta_point, opt.current_point)
        axpy!(opt.current_point, step_size, opt.next_step_direction, n)
        @assert opt.constraint_function!(opt.current_point)

        # Compute delta point and step length.
        delta!(opt.delta_point, opt.current_point, n)
        step_length = sqrt(norm2(opt.delta_point))
        opt.last_step_length[] = step_length

        # Update objective value and delta.
        opt.delta_objective_value[] =
            objective_value - opt.current_objective_value[]
        opt.current_objective_value[] = objective_value

        # Update gradient and delta.
        copy!(opt.delta_gradient, opt.current_gradient)
        opt.gradient_function!(opt.current_gradient, opt.current_point)
        delta!(opt.delta_gradient, opt.current_gradient, n)

        # If gradient is zero or non-finite, terminate.
        inv_gradient_norm = inv_norm(opt.current_gradient)
        if !isfinite(inv_gradient_norm)
            opt.has_terminated[] = true
            return opt
        end

        # Store delta point and delta gradient in history.
        c = Base.srem_int(opt.iteration_count[] - 1, m) + 1 # cyclic index
        copyto!(view(opt._delta_point_history, :, c), opt.delta_point)
        copyto!(view(opt._delta_gradient_history, :, c), opt.delta_gradient)

        # Compute delta overlap and store in rho.
        delta_overlap = dot(opt.delta_point, opt.delta_gradient, n)
        @inbounds opt._rho[c] = inv(delta_overlap)

        # Update history count.
        hist_count = min(opt._history_count[] + 1, m)
        hist_end = opt.iteration_count[]
        hist_begin = hist_end - hist_count + 1
        opt._history_count[] = hist_count

        # Compute next step direction starting from current gradient.
        copy!(opt.next_step_direction, opt.current_gradient)

        # Apply forward L-BFGS correction.
        for iter = hist_end:-1:hist_begin
            c = Base.srem_int(iter - 1, m) + 1
            @inbounds alpha = opt._rho[c] * dot(opt.next_step_direction,
                view(opt._delta_point_history, :, c), n)
            @inbounds opt._alpha[c] = alpha
            axpy!(opt.next_step_direction,
                alpha, view(opt._delta_gradient_history, :, c), n)
        end

        # Compute natural step size.
        scale!(opt.next_step_direction,
            delta_overlap / norm2(opt.delta_gradient))

        # Apply backward L-BFGS correction.
        for iter = hist_begin:hist_end
            c = Base.srem_int(iter - 1, m) + 1
            @inbounds beta = opt._alpha[c] - opt._rho[c] * dot(
                opt.next_step_direction,
                view(opt._delta_gradient_history, :, c), n)
            axpy!(opt.next_step_direction,
                beta, view(opt._delta_point_history, :, c), n)
        end

        # Verify that L-BFGS step direction is a descent direction.
        gradient_overlap = dot(
            negate!(opt.next_step_direction, n), opt.current_gradient, n)

        # If not, reset step direction to negative gradient.
        if !isfinite(gradient_overlap)
            opt.has_terminated[] = true
        elseif gradient_overlap >= _zero
            scale!(opt.next_step_direction,
                -step_length * inv_gradient_norm, opt.current_gradient, n)
        end
    end
    return opt
end


#=

########################################################## UNSAFE LINEAR ALGEBRA

@inline half(::Type{T}) where {T} = one(T) / (one(T) + one(T))

@inline function normalize!(x::AbstractArray{T,N}) where {T,N}
    a = inv(norm(x))
    @simd ivdep for i = 1 : length(x)
        @inbounds x[i] *= a
    end
    return x
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

@inline linear_view(x::Array{T,N}) where {T,N} =
    reshape(view(x, ntuple(_ -> Colon(), N)...), length(x))

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
    has_terminated::Array{Bool,0}
    current_point::Array{T,N}
    current_objective_value::Array{T,0}
    current_gradient::Array{T,N}
    delta_point::Array{T,N}
    delta_gradient::Array{T,N}
    last_step_length::Array{T,0}
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
                       initial_step_length::T) where {F,G,T,N}
    return BFGSOptimizer(
        objective_function, gradient_function!,
        NULL_CONSTRAINT, initial_point, initial_step_length)
end

function BFGSOptimizer(objective_function::F,
                       gradient_function!::G,
                       constraint_function!::C,
                       initial_point::Array{T,N},
                       initial_step_length::T) where {F,G,C,T,N}
    iteration_count = fill(0)
    has_terminated = fill(false)
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
    last_step_length = fill(initial_step_length)
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
        has_terminated,
        current_point,
        current_objective_value,
        current_gradient,
        delta_point,
        delta_gradient,
        last_step_length,
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
        fill(opt.last_step_length[]),
        fill(opt.last_step_type[]),
        approximate_inverse_hessian,
        next_step_direction,
        _scratch_space,
        _gradient_line_search_functor,
        _bfgs_line_search_functor)
end

function update_inverse_hessian!(
        inv_hess::Matrix{T}, step_length::T, step_direction::Array{T,N},
        delta_gradient::Array{T,N}, scratch_space::Array{T,N}) where {T,N}

    n = length(step_direction)
    @assert n == length(delta_gradient)
    @assert n == length(scratch_space)
    @assert (n, n) == size(inv_hess)

    overlap = dot(step_direction, delta_gradient, n)
    scalar_mul!(step_direction, inv(overlap))
    mul!(linear_view(scratch_space), inv_hess, linear_view(delta_gradient))
    delta_norm = step_length * overlap + dot(delta_gradient, scratch_space, n)

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

    if !opt.has_terminated[]

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
        step_length = opt.last_step_length[]

        # Launch line search in raw (-gradient) direction.
        grad_norm = norm(gradient)
        grad_step_length, grad_obj = quadratic_line_search(
            opt._gradient_line_search_functor,
            opt.current_objective_value[],
            step_length / grad_norm)

        # Launch line search in BFGS (Hessian)^-1 * (-gradient) direction.
        bfgs_norm = norm(bfgs_direction)
        bfgs_step_length, bfgs_obj = quadratic_line_search(
            opt._bfgs_line_search_functor,
            opt.current_objective_value[],
            step_length / bfgs_norm)

        if bfgs_obj < opt.current_objective_value[] && !(bfgs_obj > grad_obj)

            # Accept BFGS step.
            opt.current_objective_value[] = bfgs_obj
            opt.last_step_length[] = bfgs_step_length * bfgs_norm
            opt.last_step_type[] = BFGSStep
            opt.iteration_count[] += 1

            # Update point and gradient.
            negate!(delta_point, point, n)
            negate!(delta_gradient, gradient, n)
            add!(point, -bfgs_step_length, bfgs_direction, n)
            constraint_success = opt.constraint_function!(point)
            @assert constraint_success
            opt.gradient_function!(gradient, point)
            add!(delta_point, point, n)
            add!(delta_gradient, gradient, n)

            # Update inverse Hessian approximation using delta_gradient.
            update_inverse_hessian!(opt.approximate_inverse_hessian,
                                    -bfgs_step_length, bfgs_direction,
                                    delta_gradient, opt._scratch_space)

            # Compute next step direction using approximate inverse Hessian.
            mul!(linear_view(bfgs_direction),
                 opt.approximate_inverse_hessian,
                 linear_view(gradient))

        elseif grad_obj < opt.current_objective_value[]

            # Accept gradient descent step.
            opt.current_objective_value[] = grad_obj
            opt.last_step_length[] = grad_step_length * grad_norm
            opt.last_step_type[] = GradientDescentStep
            opt.iteration_count[] += 1

            # Update point and gradient.
            negate!(delta_point, point, n)
            negate!(delta_gradient, gradient, n)
            add!(point, -grad_step_length, gradient, n)
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
            opt.has_terminated[] = true
        end

    end
    return opt
end

############################################################## OPTIMIZER TESTING

function run_and_test!(opt)

    history = [deepcopy(opt)]
    while !opt.has_terminated[]
        step!(opt)
        push!(history, deepcopy(opt))
    end

    # None of the optimizers should have converged, except the last one.
    for i = 1 : length(history) - 1
        @assert !history[i].has_terminated[]
    end
    @assert history[end].has_terminated[]

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


=#


end # module DZOptimization
