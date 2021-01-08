baremodule ExampleFunctions

using Base: +, -, *, AbstractVector, @inbounds

function rosenbrock_objective(v::AbstractVector{T}) where {T}
    @inbounds x, y = v[1], v[2]
    t1 = 1 - x
    t2 = y - x * x
    t1 * t1 + 100 * (t2 * t2)
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

end # baremodule ExampleFunctions
