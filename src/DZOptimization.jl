module DZOptimization

using KernelAbstractions: get_backend
using LinearAlgebra: axpy!, dot

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
    current_point::A,
    current_objective_value::T,
    current_gradient::A,
    step_direction::A,
    overlap::T,
) where {C,F,G,T,A<:AbstractArray{T}}

    point_axes = axes(current_point)
    @assert point_axes == axes(current_gradient)
    @assert point_axes == axes(step_direction)

    backend = get_backend(current_point)
    @assert backend == get_backend(current_gradient)
    @assert backend == get_backend(step_direction)

    trial_point = similar(current_point)
    @assert point_axes == axes(trial_point)
    @assert backend == get_backend(trial_point)

    trial_gradient = similar(current_gradient)
    @assert point_axes == axes(trial_gradient)
    @assert backend == get_backend(trial_gradient)

    return LineSearchEvaluator{C,F,G,T,A}(
        constraint_function!, objective_function, gradient_function!,
        current_point, fill(current_objective_value), current_gradient,
        step_direction, fill(overlap),
        trial_point, Array{T,0}(undef), trial_gradient,
        Array{T,0}(undef), Array{T,0}(undef))
end


function LineSearchEvaluator(
    objective_function::F,
    gradient_function!::G,
    current_point::A,
    current_gradient::A,
    step_direction::A,
) where {F,G,T,A<:AbstractArray{T}}

    point_axes = axes(current_point)
    @assert point_axes == axes(current_gradient)
    @assert point_axes == axes(step_direction)

    backend = get_backend(current_point)
    @assert backend == get_backend(current_gradient)
    @assert backend == get_backend(step_direction)

    trial_point = similar(current_point)
    @assert point_axes == axes(trial_point)
    @assert backend == get_backend(trial_point)

    trial_gradient = similar(current_gradient)
    @assert point_axes == axes(trial_gradient)
    @assert backend == get_backend(trial_gradient)

    return LineSearchEvaluator{Nothing,F,G,T,A}(
        nothing, objective_function, gradient_function!,
        current_point, Array{T,0}(undef), current_gradient,
        step_direction, Array{T,0}(undef),
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

include("ExampleFunctions.jl")

end # module DZOptimization
