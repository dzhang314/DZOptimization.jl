module DZOptimization

using KernelAbstractions: get_backend
using LinearAlgebra: axpby!, axpy!, dot, norm

################################################################################


export LineSearchEvaluator


struct LineSearchEvaluator{C,F,G,T,A}
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
) where {C,F,G,T,A<:AbstractArray{T}}

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

    return LineSearchEvaluator{C,F,G,T,A}(
        constraint_function!, objective_function, gradient_function!,
        initial_point, fill(initial_objective_value), initial_gradient,
        step_direction, fill(overlap),
        trial_point, Array{T,0}(undef), trial_gradient,
        Array{T,0}(undef), Array{T,0}(undef))
end


function (lse::LineSearchEvaluator{C,F,G,T,A})(
    step_size::T,
    compute_gradient::Bool,
) where {C,F,G,T,A}
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


export step!


function step! end


################################################################################


export AdGDOptimizer


struct AdGDOptimizer{C,F,G,T,A}

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

    current_lambda::Array{T,0}
    previous_lambda::Array{T,0}

end


function AdGDOptimizer(
    constraint_function!::C,
    objective_function::F,
    gradient_function!::G,
    initial_point::A,
    initial_objective_value::T,
    initial_gradient::A,
    initial_step_length::T,
) where {C,F,G,T,A<:AbstractArray{T}}

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

    initial_lambda = abs(initial_step_length) / norm(initial_gradient)

    return AdGDOptimizer{C,F,G,T,A}(
        constraint_function!, objective_function, gradient_function!,
        fill(false), fill(0),
        initial_point, delta_point,
        fill(initial_objective_value), fill(_zero),
        initial_gradient, delta_gradient,
        fill(initial_lambda), fill(initial_lambda))
end


function AdGDOptimizer(
    constraint_function!::C,
    objective_function::F,
    gradient_function!::G,
    initial_point::A,
    initial_step_length::T,
) where {C,F,G,T,A<:AbstractArray{T}}

    point_axes = axes(initial_point)
    backend = get_backend(initial_point)

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


function step!(opt::AdGDOptimizer{C,F,G,T,A}) where {C,F,G,T,A}

    if opt.is_stuck[]
        return opt
    end

    _one = one(T)
    _two = _one + _one
    _half = inv(_two)

    previous_lambda = opt.previous_lambda[]
    current_lambda = opt.current_lambda[]
    if opt.iteration_count[] == 0
        next_lambda = current_lambda
    else
        theta = current_lambda / previous_lambda
        next_lambda = min(sqrt(_one + theta) * current_lambda,
            _half * norm(opt.delta_point) / norm(opt.delta_gradient))
    end
    opt.previous_lambda[] = current_lambda
    opt.current_lambda[] = next_lambda

    copy!(opt.delta_point, opt.current_point)
    while true
        axpy!(-next_lambda, opt.current_gradient, opt.current_point)
        if isequal(opt.current_point, opt.delta_point)
            opt.is_stuck[] = true
            return opt
        end
        if (isnothing(opt.constraint_function!) ||
            opt.constraint_function!(opt.current_point))
            next_objective_value = opt.objective_function(opt.current_point)
            if next_objective_value < opt.current_objective_value[]
                opt.delta_objective_value[] =
                    next_objective_value - opt.current_objective_value[]
                opt.current_objective_value[] = next_objective_value
                break
            end
        end
        copy!(opt.current_point, opt.delta_point)
        next_lambda *= _half
    end
    axpby!(_one, opt.current_point, -_one, opt.delta_point)

    copy!(opt.delta_gradient, opt.current_gradient)
    opt.gradient_function!(opt.current_gradient, opt.current_point)
    axpby!(_one, opt.current_gradient, -_one, opt.delta_gradient)

    opt.iteration_count[] += 1
    return opt
end


################################################################################

include("ExampleFunctions.jl")

end # module DZOptimization
