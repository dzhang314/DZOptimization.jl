module DZOptimization

using KernelAbstractions: get_backend
using LinearAlgebra: axpby!, axpy!, dot, norm, rmul!

################################################################################


export LineSearchEvaluator


struct LineSearchEvaluator{T,A,C,F,G}
    constraint_function!::C
    objective_function::F
    gradient_function!::G
    current_point::A
    current_objective_value::Array{T,0}
    current_gradient::A
    step_direction::A
    overlap::Array{T,0}
    trial_point::A
    trial_objective_value::Array{T,0}
    trial_gradient::A
    improvement_ratio::Array{T,0}
    slope_ratio::Array{T,0}
end


function LineSearchEvaluator(
    constraint_function!::C,
    objective_function::F,
    gradient_function!::G,
    initial_point::A,
    initial_objective_value::T,
    initial_gradient::A,
    step_direction::A,
    overlap::T,
) where {T,A<:AbstractArray{T},C,F,G}

    point_axes = axes(initial_point)
    @assert point_axes == axes(initial_gradient)
    @assert point_axes == axes(step_direction)

    backend = get_backend(initial_point)
    @assert backend == get_backend(initial_gradient)
    @assert backend == get_backend(step_direction)

    trial_point = similar(initial_point)
    @assert point_axes == axes(trial_point)
    @assert backend == get_backend(trial_point)

    trial_gradient = similar(initial_gradient)
    @assert point_axes == axes(trial_gradient)
    @assert backend == get_backend(trial_gradient)

    return LineSearchEvaluator{T,A,C,F,G}(
        constraint_function!, objective_function, gradient_function!,
        initial_point, fill(initial_objective_value), initial_gradient,
        step_direction, fill(overlap),
        trial_point, Array{T,0}(undef), trial_gradient,
        Array{T,0}(undef), Array{T,0}(undef))
end


function (lse::LineSearchEvaluator{T,A,C,F,G})(
    step_size::T,
    compute_gradient::Bool,
) where {T,A<:AbstractArray{T},C,F,G}
    copy!(lse.trial_point, lse.current_point)
    axpy!(step_size, lse.step_direction, lse.trial_point)
    if !isnothing(lse.constraint_function!)
        if !lse.constraint_function!(lse.trial_point)
            result = typemax(T)
            lse.trial_objective_value[] = result
            lse.improvement_ratio[] = typemin(T)
            lse.slope_ratio[] = result
            return result
        end
    end
    f_new = lse.objective_function(lse.trial_point)
    lse.trial_objective_value[] = f_new
    f_old = lse.current_objective_value[]
    overlap = lse.overlap[]
    lse.improvement_ratio[] = (f_new - f_old) / (step_size * overlap)
    if compute_gradient
        @assert !isnothing(lse.gradient_function!)
        lse.gradient_function!(lse.trial_gradient, lse.trial_point)
        trial_overlap = dot(lse.trial_gradient, lse.step_direction)
        lse.slope_ratio[] = trial_overlap / overlap
    end
    return f_new
end


################################################################################


export AbstractOptimizer, step!


abstract type AbstractOptimizer{T,A} end


function step! end


function take_backtracking_step!(
    opt::AbstractOptimizer{T,A},
    step_size::T,
    step_direction::A,
) where {T,A}

    _one = one(T)
    _two = _one + _one
    _half = inv(_two)

    # Make a copy of the current point in case we need to backtrack.
    copy!(opt.delta_point, opt.current_point)

    # Begin backtracking loop.
    while true

        # Take a step of the specified size in the specified direction.
        axpy!(step_size, step_direction, opt.current_point)

        # If the step was so small that we didn't change any
        # coordinate of the current point, then we are stuck.
        if isequal(opt.current_point, opt.delta_point)
            opt.is_stuck[] = true
            return opt
        end

        # If projection into the feasible region succeeds...
        if (isnothing(opt.constraint_function!) ||
            opt.constraint_function!(opt.current_point))

            # ...and the objective value strictly decreases...
            next_objective_value = opt.objective_function(opt.current_point)
            if next_objective_value < opt.current_objective_value[]

                # ...then we record the decrease and accept the step.
                opt.delta_objective_value[] =
                    next_objective_value - opt.current_objective_value[]
                opt.current_objective_value[] = next_objective_value
                axpby!(_one, opt.current_point, -_one, opt.delta_point)
                return opt
            end
        end

        # Otherwise, we backtrack, halve the step size, and try again.
        copy!(opt.current_point, opt.delta_point)
        step_size *= _half
    end
end


###################################################### ADAPTIVE GRADIENT DESCENT


export AdGDOptimizer


"""
`AdGDOptimizer` implements the adaptive gradient descent algorithm (AdGD)
proposed by Yura Malitsky and Konstantin Mishchenko in the ICML 2020 paper

    [MM20] "Adaptive Gradient Descent without Descent"
    https://arxiv.org/abs/1910.09529
    https://dl.acm.org/doi/10.5555/3524938.3525560

and further improved by the same authors in the NeurIPS 2024 paper

    [MM24] "Adaptive Proximal Gradient Method for Convex Optimization"
    https://arxiv.org/abs/2308.02261
    https://dl.acm.org/doi/10.5555/3737916.3741109

The algorithm implemented here is Algorithm 1 from the second paper [MM24].
"""
struct AdGDOptimizer{T,A,C,F,G} <: AbstractOptimizer{T,A}

    constraint_function!::C
    objective_function::F
    gradient_function!::G

    is_stuck::Array{Bool,0}
    iteration_count::Array{Int,0}

    current_point::A
    delta_point::A
    current_objective_value::Array{T,0}
    delta_objective_value::Array{T,0}
    current_gradient::A
    delta_gradient::A

    current_step_size::Array{T,0}
    previous_step_size::Array{T,0}

end


function AdGDOptimizer(
    constraint_function!::C,
    objective_function::F,
    gradient_function!::G,
    initial_point::A,
    initial_objective_value::T,
    initial_gradient::A,
    initial_step_length::T,
) where {T,A<:AbstractArray{T},C,F,G}

    _zero = zero(T)

    point_axes = axes(initial_point)
    @assert point_axes == axes(initial_gradient)

    backend = get_backend(initial_point)
    @assert backend == get_backend(initial_gradient)

    delta_point = similar(initial_point)
    @assert point_axes == axes(delta_point)
    @assert backend == get_backend(delta_point)
    fill!(delta_point, _zero)

    delta_gradient = similar(initial_gradient)
    @assert point_axes == axes(delta_gradient)
    @assert backend == get_backend(delta_gradient)
    fill!(delta_gradient, _zero)

    @assert initial_step_length > _zero
    initial_gradient_norm = norm(initial_gradient)
    is_stuck = iszero(initial_gradient_norm)
    initial_step_size =
        is_stuck ? _zero : initial_step_length / initial_gradient_norm

    return AdGDOptimizer{T,A,C,F,G}(
        constraint_function!, objective_function, gradient_function!,
        fill(is_stuck), fill(0),
        initial_point, delta_point,
        fill(initial_objective_value), fill(_zero),
        initial_gradient, delta_gradient,
        fill(initial_step_size), fill(initial_step_size))
end


function AdGDOptimizer(
    constraint_function!::C,
    objective_function::F,
    gradient_function!::G,
    initial_point::A,
    initial_step_length::T,
) where {T,A<:AbstractArray{T},C,F,G}

    point_axes = axes(initial_point)
    backend = get_backend(initial_point)

    if !isnothing(constraint_function!)
        @assert constraint_function!(initial_point)
    end

    initial_objective_value = objective_function(initial_point)

    initial_gradient = similar(initial_point)
    @assert point_axes == axes(initial_gradient)
    @assert backend == get_backend(initial_gradient)
    gradient_function!(initial_gradient, initial_point)

    return AdGDOptimizer(
        constraint_function!, objective_function, gradient_function!,
        initial_point, initial_objective_value, initial_gradient,
        initial_step_length)
end


function step!(opt::AdGDOptimizer{T,A,C,F,G}) where {T,A,C,F,G}

    if opt.is_stuck[]
        return opt
    end

    _one = one(T)
    _two = _one + _one
    _half = inv(_two)
    _inv_sqrt_two = sqrt(_half)

    previous_step_size = opt.previous_step_size[]
    current_step_size = opt.current_step_size[]
    next_step_size = current_step_size
    if opt.iteration_count[] > 0
        @assert !iszero(previous_step_size)
        theta = current_step_size / previous_step_size
        next_step_size *= sqrt(_one + theta)
        delta_gradient_norm = norm(opt.delta_gradient)
        if !iszero(delta_gradient_norm)
            inv_L = norm(opt.delta_point) / delta_gradient_norm
            next_step_size = min(next_step_size, _inv_sqrt_two * inv_L)
        end
    end
    opt.previous_step_size[] = current_step_size
    opt.current_step_size[] = next_step_size

    take_backtracking_step!(opt, -next_step_size, opt.current_gradient)
    if opt.is_stuck[]
        return opt
    end

    copy!(opt.delta_gradient, opt.current_gradient)
    opt.gradient_function!(opt.current_gradient, opt.current_point)
    axpby!(_one, opt.current_gradient, -_one, opt.delta_gradient)

    opt.iteration_count[] += 1
    return opt
end


######################################################################### L-BFGS


export LBFGSOptimizer


struct LBFGSOptimizer{T,A,C,F,G} <: AbstractOptimizer{T,A}

    constraint_function!::C
    objective_function::F
    gradient_function!::G

    is_stuck::Array{Bool,0}
    iteration_count::Array{Int,0}

    current_point::A
    delta_point::A
    current_objective_value::Array{T,0}
    delta_objective_value::Array{T,0}
    current_gradient::A
    delta_gradient::A

    step_direction::A
    history_length::Int
    delta_point_history::Vector{A}
    delta_gradient_history::Vector{A}
    alpha_history::Vector{T}
    rho_history::Vector{T}

end


function LBFGSOptimizer(
    constraint_function!::C,
    objective_function::F,
    gradient_function!::G,
    initial_point::A,
    initial_objective_value::T,
    initial_gradient::A,
    initial_step_length::T,
    history_length::Int,
) where {T,A<:AbstractArray{T},C,F,G}

    _zero = zero(T)

    point_axes = axes(initial_point)
    @assert point_axes == axes(initial_gradient)

    backend = get_backend(initial_point)
    @assert backend == get_backend(initial_gradient)

    delta_point = similar(initial_point)
    @assert point_axes == axes(delta_point)
    @assert backend == get_backend(delta_point)
    fill!(delta_point, _zero)

    delta_gradient = similar(initial_gradient)
    @assert point_axes == axes(delta_gradient)
    @assert backend == get_backend(delta_gradient)
    fill!(delta_gradient, _zero)

    step_direction = similar(initial_gradient)
    @assert point_axes == axes(step_direction)
    @assert backend == get_backend(step_direction)

    @assert initial_step_length > _zero
    initial_gradient_norm = norm(initial_gradient)
    is_stuck = iszero(initial_gradient_norm)
    if is_stuck
        fill!(step_direction, _zero)
    else
        copy!(step_direction, initial_gradient)
        rmul!(step_direction, -initial_step_length / initial_gradient_norm)
    end

    return LBFGSOptimizer{T,A,C,F,G}(
        constraint_function!, objective_function, gradient_function!,
        fill(is_stuck), fill(0),
        initial_point, delta_point,
        fill(initial_objective_value), fill(_zero),
        initial_gradient, delta_gradient,
        step_direction, history_length, A[], A[], T[], T[])
end


function LBFGSOptimizer(
    constraint_function!::C,
    objective_function::F,
    gradient_function!::G,
    initial_point::A,
    initial_step_length::T,
    history_length::Int,
) where {T,A<:AbstractArray{T},C,F,G}

    point_axes = axes(initial_point)
    backend = get_backend(initial_point)

    if !isnothing(constraint_function!)
        @assert constraint_function!(initial_point)
    end

    initial_objective_value = objective_function(initial_point)

    initial_gradient = similar(initial_point)
    @assert point_axes == axes(initial_gradient)
    @assert backend == get_backend(initial_gradient)
    gradient_function!(initial_gradient, initial_point)

    return LBFGSOptimizer(
        constraint_function!, objective_function, gradient_function!,
        initial_point, initial_objective_value, initial_gradient,
        initial_step_length, history_length)
end


function compute_lbfgs_step_direction!(
    step_direction::A,
    gradient::A,
    s::Vector{A},
    y::Vector{A},
    alpha::Vector{T},
    rho::Vector{T},
) where {T,A}
    copy!(step_direction, gradient)
    for i in eachindex(s, y, alpha, rho)
        alpha[i] = dot(s[i], step_direction) / rho[i]
        axpy!(-alpha[i], y[i], step_direction)
    end
    if !isempty(eachindex(s, y, alpha, rho))
        rmul!(step_direction, -rho[1] / dot(y[1], y[1]))
    end
    for i in Iterators.reverse(eachindex(s, y, alpha, rho))
        beta = dot(y[i], step_direction) / rho[i]
        axpy!(-(alpha[i] + beta), s[i], step_direction)
    end
    return step_direction
end


function step!(opt::LBFGSOptimizer{T,A,C,F,G}) where {T,A,C,F,G}

    if opt.is_stuck[]
        return opt
    end

    _zero = zero(T)
    _one = one(T)

    if opt.iteration_count[] > 0
        compute_lbfgs_step_direction!(
            opt.step_direction,
            opt.current_gradient,
            opt.delta_point_history,
            opt.delta_gradient_history,
            opt.alpha_history,
            opt.rho_history)
    end

    take_backtracking_step!(opt, _one, opt.step_direction)
    if opt.is_stuck[]
        return opt
    end

    copy!(opt.delta_gradient, opt.current_gradient)
    opt.gradient_function!(opt.current_gradient, opt.current_point)
    axpby!(_one, opt.current_gradient, -_one, opt.delta_gradient)

    if length(opt.delta_point_history) < opt.history_length
        pushfirst!(opt.delta_point_history, copy(opt.delta_point))
    else
        delta_point = pop!(opt.delta_point_history)
        copy!(delta_point, opt.delta_point)
        pushfirst!(opt.delta_point_history, delta_point)
    end

    if length(opt.delta_gradient_history) < opt.history_length
        pushfirst!(opt.delta_gradient_history, copy(opt.delta_gradient))
    else
        delta_gradient = pop!(opt.delta_gradient_history)
        copy!(delta_gradient, opt.delta_gradient)
        pushfirst!(opt.delta_gradient_history, delta_gradient)
    end

    if length(opt.alpha_history) < opt.history_length
        push!(opt.alpha_history, _zero)
    end

    if length(opt.rho_history) >= opt.history_length
        pop!(opt.rho_history)
    end
    pushfirst!(opt.rho_history, dot(opt.delta_point, opt.delta_gradient))

    opt.iteration_count[] += 1
    return opt
end


################################################################################

include("ExampleFunctions.jl")

end # module DZOptimization
