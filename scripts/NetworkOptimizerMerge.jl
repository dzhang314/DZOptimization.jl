using Base.Threads
using DZOptimization.SortingNetworks
using JLD2


function load_all(dir::AbstractString, prefix::AbstractString)
    paths = [path for path in readdir(dir; join=true, sort=false) if
             startswith(basename(path), prefix) &&
             endswith(basename(path), ".jld2") && isfile(path)]
    result = similar(paths, Any)
    @threads :dynamic for i in eachindex(result, paths)
        result[i] = load_object(paths[i])
    end
    return identity.(result) # manually trigger type inference
end


@assert isdir(ARGS[1])
for prefix in [
    "SearchW2", "SearchW3", "SearchW4", "SearchW5",
    "SearchK2", "SearchK3", "SearchK4", "SearchK5",
    "SearchS2", "SearchS3", "SearchS4", "SearchS5",
    "SearchC2", "SearchC3", "SearchC4", "SearchC5"]

    println("Loading ", prefix, " data...")
    flush(stdout)
    optimizers = load_all(ARGS[1], prefix)

    println("Merging (maximal)...")
    flush(stdout)
    max_opt = combine(optimizers;
        keep_failing_networks=:all, keep_test_cases=:maximal, verbose=true)

    println("Finished merging. Checking validity...")
    flush(stdout)
    assert_valid(max_opt)

    println("Saving (maximal)...")
    flush(stdout)
    save_object("$(prefix)-Maximal.jld2", max_opt)

    println("Merging (minimal)...")
    flush(stdout)
    min_opt = combine(optimizers;
        keep_failing_networks=:none, keep_test_cases=:covering, verbose=true)

    println("Finished merging. Checking validity...")
    flush(stdout)
    assert_valid(min_opt)

    println("Saving (minimal)...")
    flush(stdout)
    save_object("$(prefix)-Minimal.jld2", min_opt)

    println("Finished merging ", prefix, ".")
    flush(stdout)
end
