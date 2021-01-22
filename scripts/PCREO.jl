using LinearAlgebra: eigvals!, Symmetric
using Random: seed!
using UUIDs: uuid4

using DZOptimization
using DZOptimization: half, normalize_columns!
using DZOptimization.ExampleFunctions:
    riesz_energy, riesz_gradient!, riesz_hessian!,
    constrain_riesz_gradient_sphere!, constrain_riesz_hessian_sphere!

using MultiFloats


function constrain_sphere!(points)
    normalize_columns!(points)
    return true
end

function spherical_riesz_gradient!(grad, points)
    riesz_gradient!(grad, points)
    constrain_riesz_gradient_sphere!(grad, points)
    return grad
end

spherical_riesz_gradient(points) =
    spherical_riesz_gradient!(similar(points), points)

function spherical_riesz_hessian(points::Matrix{T}) where {T}
    unconstrained_grad = riesz_gradient!(similar(points), points)
    hess = Array{T,4}(undef, size(points)..., size(points)...)
    riesz_hessian!(hess, points)
    constrain_riesz_hessian_sphere!(hess, points, unconstrained_grad)
    return reshape(hess, length(points), length(points))
end


const F = typeof(riesz_energy)
const G = typeof(spherical_riesz_gradient!)
const C = typeof(constrain_sphere!)
const N = 2


function run!(opt::BFGSOptimizer{F,G,C,Float64,N})
    while !opt.has_converged[]
        step!(opt)
    end
    return opt
end

function run!(opt::BFGSOptimizer{F,G,C,T,N}) where {T}
    println(opt.iteration_count[], '\t',
            opt.current_objective_value[])
    while !opt.has_converged[]
        step!(opt)
        step!(opt)
        step!(opt)
        step!(opt)
        step!(opt)
        println(opt.iteration_count[], '\t',
                opt.current_objective_value[])
    end
    return opt
end

function refine(points::Matrix{T}) where {T}
    return run!(BFGSOptimizer(riesz_energy, spherical_riesz_gradient!,
                              constrain_sphere!, points, T(0.01)))
end

function refine(::Type{T}, opt::BFGSOptimizer{F,G,C,U,N}) where {T,U}
    return run!(BFGSOptimizer(T, opt))
end

function generate_point_configuration(dimension::Int, num_points::Int)
    initial_points = normalize_columns!(randn(dimension, num_points))
    opt1 = refine(initial_points)
    opt2 = refine(Float64x2, opt1)
    opt3 = refine(Float64x3, opt2)
    return (Float64(opt3.current_objective_value[]),
            Float64.(opt3.current_point), initial_points)
end


spherical_riesz_gradient_norm(points) =
    maximum(abs.(spherical_riesz_gradient(points)))

function symmetrize!(mat::Matrix{T}) where {T}
    m, n = size(mat)
    @assert m == n
    @inbounds for i = 1 : n-1
        @simd ivdep for j = i+1 : n
            sym = half(T) * (mat[i, j] + mat[j, i])
            mat[i, j] = mat[j, i] = sym
        end
    end
    return mat
end

function spherical_riesz_hessian_spectral_gap(points)
    dim, num_points = size(points)
    hess = symmetrize!(spherical_riesz_hessian(points))
    vals = eigvals!(hess)
    num_expected_zeros = div(dim * (dim - 1), 2) + num_points
    expected_zero_vals = vals[1:num_expected_zeros]
    expected_nonzero_vals = vals[num_expected_zeros+1:end]
    @assert all(!signbit, expected_nonzero_vals)
    return (maximum(abs.(expected_zero_vals)) /
            minimum(expected_nonzero_vals))
end


function distances(points::Matrix{T}) where {T}
    dimension, num_points = size(points)
    num_pairs = div(num_points * (num_points - 1), 2)
    result = Vector{T}(undef, num_pairs)
    p = 0
    for i = 1 : num_points-1
        for j = i+1 : num_points
            dist_sq = zero(T)
            @simd ivdep for k = 1 : dimension
                @inbounds dist_sq += abs2(points[k,i] - points[k,j])
            end
            @inbounds result[p += 1] = DZOptimization.unsafe_sqrt(dist_sq)
        end
    end
    return result
end


const QCONVEX_PATH = "C:\\Programs\\qhull-2020.2\\bin\\qconvex.exe"

function convex_hull_facets(points)
    dim, num_points = size(points)
    buffer = IOBuffer()
    process = open(`$QCONVEX_PATH i`, buffer, write=true)
    println(process, dim)
    println(process, num_points)
    for j = 1 : num_points
        for i = 1 : dim
            print(process, ' ', points[i,j])
        end
        println(process)
    end
    close(process)
    while process_running(process)
        sleep(0.001)
    end
    first = true
    num_facets = 0
    result = Vector{Int}[]
    seek(buffer, 0)
    for line in eachline(buffer)
        if first
            num_facets = parse(Int, line)
            first = false
        else
            push!(result, [parse(Int, s) + 1 for s in split(line)])
        end
    end
    @assert num_facets == length(result)
    return result
end


function generate_and_save_point_configuration(dimension::Int, num_points::Int)
    energy, points, initial = generate_point_configuration(
        dimension, num_points)
    grad_norm = spherical_riesz_gradient_norm(points)
    spectral_gap = spherical_riesz_hessian_spectral_gap(points)
    success = (grad_norm < 1.0e-10) && (spectral_gap < 1.0e-10)
    facets = convex_hull_facets(points)
    open("$(ifelse(success, "PCREO", "FAIL"))-$(lpad(dimension, 2, '0'))-" *
         "$(lpad(num_points, 4, '0'))-$(uuid4()).csv", "w+") do io
        println(io, dimension)
        println(io, num_points)
        println(io, energy)
        println(io)
        for j = 1 : num_points
            for i = 1 : dimension
                if i > 1
                    print(io, ", ")
                end
                print(io, points[i,j])
            end
            println(io)
        end
        println(io)
        for facet in facets
            for i = 1 : length(facet)
                if i > 1
                    print(io, ", ")
                end
                print(io, facet[i])
            end
            println(io)
        end
        println(io)
        for j = 1 : num_points
            for i = 1 : dimension
                if i > 1
                    print(io, ", ")
                end
                print(io, initial[i,j])
            end
            println(io)
        end
    end
end


function main()
    while true
        for num_points = 50 : 199
            generate_and_save_point_configuration(3, num_points)
        end
    end
end

main()
