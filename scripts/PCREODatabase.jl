using DZOptimization
using LinearAlgebra: cross, det
using NearestNeighbors

const PCREO_FILENAME_REGEX = r"^PCREO-([0-9]{2})-([0-9]{4})-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.csv$"
const PCREO_DIRECTORY = "D:\\Data\\PCREO"
const DATABASE_DIRECTORY = "D:\\Data\\PCREODatabase"


struct PCREORecord
    dimension::Int
    num_points::Int
    energy::Float64
    points::Matrix{Float64}
    facets::Vector{Vector{Int}}
    initial::Matrix{Float64}
end


function read_pcreo_file(path)
    filename = basename(path)
    m = match(PCREO_FILENAME_REGEX, filename)
    @assert !isnothing(m)
    dimension = parse(Int, m[1])
    num_points = parse(Int, m[2])
    uuid = m[3]
    data = split(read(path, String), "\n\n")
    @assert length(data) == 4
    header = split(data[1])
    @assert length(header) == 3
    @assert dimension == parse(Int, header[1])
    @assert num_points == parse(Int, header[2])
    energy = parse(Float64, header[3])
    points = hcat([[parse(Float64, strip(entry))
                    for entry in split(line, ',')]
                   for line in split(strip(data[2]), '\n')]...)
    @assert (dimension, num_points) == size(points)
    facets = [[parse(Int, strip(entry))
               for entry in split(line, ',')]
              for line in split(strip(data[3]), '\n')]
    initial = hcat([[parse(Float64, strip(entry))
                     for entry in split(line, ',')]
                    for line in split(strip(data[4]), '\n')]...)
    @assert (dimension, num_points) == size(initial)
    return PCREORecord(dimension, num_points, energy,
                       points, facets, initial)
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


function labeled_distances(points::Matrix{T}) where {T}
    dimension, num_points = size(points)
    num_pairs = div(num_points * (num_points - 1), 2)
    result = Vector{Tuple{T,Int,Int}}(undef, num_pairs)
    p = 0
    for i = 1 : num_points-1
        for j = i+1 : num_points
            dist_sq = zero(T)
            @simd ivdep for k = 1 : dimension
                @inbounds dist_sq += abs2(points[k,i] - points[k,j])
            end
            @inbounds result[p += 1] = (DZOptimization.unsafe_sqrt(dist_sq), i, j)
        end
    end
    return result
end


function bucket_by_first(items::Vector{T}, epsilon) where {T}
    result = Vector{T}[]
    if length(items) == 0
        return result
    end
    push!(result, [items[1]])
    for i = 2 : length(items)
        if abs(items[i][1] - result[end][end][1]) <= epsilon
            push!(result[end], items[i])
        else
            push!(result, [items[i]])
        end
    end
    return result
end


middle(x::Vector) = x[div(length(x) + 1, 2)]


positive_transformation_matrix(u1, v1, u2, v2) =
    hcat(u2, v2, cross(u2, v2)) * inv(hcat(u1, v1, cross(u1, v1)))


negative_transformation_matrix(u1, v1, u2, v2) =
    hcat(u2, v2, cross(u2, v2)) * inv(hcat(u1, v1, cross(v1, u1)))


function isometric(points::Matrix{Float64}, tree::KDTree)
    inds, dists = knn(tree, points, 1)
    @assert all(==(1) ∘ length, inds)
    @assert all(==(1) ∘ length, dists)
    return allunique(first.(inds)) && all(<(1.0e-14) ∘ first, dists)
end


function isometric(a::PCREORecord, b::PCREORecord)
    if (a.dimension != b.dimension) || (a.num_points != b.num_points)
        return false
    end
    if !isapprox(a.energy, b.energy; rtol=1.0e-14)
        return false
    end
    if sort!(length.(a.facets)) != sort!(length.(b.facets))
        return false
    end
    a_buckets = bucket_by_first(sort!(labeled_distances(a.points)), 1.0e-14)
    b_buckets = bucket_by_first(sort!(labeled_distances(b.points)), 1.0e-14)
    if length.(a_buckets) != length.(b_buckets)
        return false
    end
    if !all(isapprox.(first.(middle.(a_buckets)),
                      first.(middle.(b_buckets)); rtol=1.0e-14))
        return false
    end
    b_tree = KDTree(b.points)
    _, i, j = middle(a_buckets[1])
    for (_, k, l) in b_buckets[1]
        mat = positive_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,k], b.points[:,l])
        @assert maximum(abs.(mat' * mat - one(mat))) < 1.0e-14
        if isometric(mat * a.points, b_tree)
            return true
        end
        mat = negative_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,k], b.points[:,l])
        @assert maximum(abs.(mat' * mat - one(mat))) < 1.0e-14
        if isometric(mat * a.points, b_tree)
            return true
        end
        mat = positive_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,l], b.points[:,k])
        @assert maximum(abs.(mat' * mat - one(mat))) < 1.0e-14
        if isometric(mat * a.points, b_tree)
            return true
        end
        mat = negative_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,l], b.points[:,k])
        @assert maximum(abs.(mat' * mat - one(mat))) < 1.0e-14
        if isometric(mat * a.points, b_tree)
            return true
        end
    end
    return false
end


function add_to_database(dataname)
    datapath = joinpath(PCREO_DIRECTORY, dataname)
    data = read_pcreo_file(datapath)
    for dirname in readdir(DATABASE_DIRECTORY)
        dirpath = joinpath(DATABASE_DIRECTORY, dirname)
        @assert isdir(dirpath)
        reppath = joinpath(dirpath, dirname * ".csv")
        @assert isfile(reppath)
        representative = read_pcreo_file(reppath)
        if isometric(data, representative)
            mv(datapath, joinpath(dirpath, dataname))
            return dirname
        end
    end
    newdirname = dataname[1:end-4]
    newdirpath = joinpath(DATABASE_DIRECTORY, newdirname)
    mkdir(newdirpath)
    mv(datapath, joinpath(newdirpath, dataname))
    return newdirname
end


function main()

    println("Checking validity of database...")
    for dirname in readdir(DATABASE_DIRECTORY)
        dirpath = joinpath(DATABASE_DIRECTORY, dirname)
        @assert isdir(dirpath)
        reppath = joinpath(dirpath, dirname * ".csv")
        @assert isfile(reppath)
        representative = read_pcreo_file(reppath)
    end

    while true
        remaining = filter(startswith("PCREO"), readdir(PCREO_DIRECTORY))
        if !isempty(remaining)
            name = rand(remaining)
            print(length(remaining), '\t', name, " => ")
            flush(stdout)
            found = add_to_database(name)
            if occursin(found, name)
                println("new")
            else
                println(found)
            end
            flush(stdout)
        end
    end

end


main()
