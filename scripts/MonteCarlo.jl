using Base.Threads: @threads
using DZOptimization.ExampleFunctions
using DZOptimization.ExampleFunctions: twice, square
using Plots
unicodeplots()


function parallel_temper!(
    energy_function::F,
    energies::AbstractMatrix{T},
    replicas::AbstractArray{T,3},
    inverse_temperatures::AbstractVector{T},
    perturbation_radii::AbstractVector{T},
    constraining_radius::T,
) where {F,T}

    _zero = zero(T)
    _one = one(T)
    _two = twice(_one)

    _fac = _two
    for _ = 1:10
        _fac = sqrt(_fac)
    end

    iteration_axis = axes(energies, 1)
    replica_axis = axes(energies, 2)
    particle_axis = axes(replicas, 1)
    xyz_axis = axes(replicas, 2)
    @assert (iteration_axis, replica_axis) == axes(energies)
    @assert (particle_axis, xyz_axis, replica_axis) == axes(replicas)
    @assert (replica_axis,) == axes(inverse_temperatures)
    @assert (replica_axis,) == axes(perturbation_radii)
    @assert length(xyz_axis) == 3
    x, y, z = xyz_axis

    @inbounds begin
        @threads for k in replica_axis
            energy = pairwise_radial_energy(
                energy_function,
                view(replicas, :, x, k),
                view(replicas, :, y, k),
                view(replicas, :, z, k))
            inv_temp = inverse_temperatures[k]
            perturbation_radius = perturbation_radii[k]
            num_accept = 0
            num_reject = 0
            for i in iteration_axis
                j = rand(particle_axis)
                x_old = replicas[j, x, k]
                y_old = replicas[j, y, k]
                z_old = replicas[j, z, k]
                x_new = x_old + perturbation_radius * randn(T)
                y_new = y_old + perturbation_radius * randn(T)
                z_new = z_old + perturbation_radius * randn(T)
                if square(x_new) + square(y_new) + square(z_new) < square(constraining_radius)
                    delta = pairwise_radial_energy_delta(
                        energy_function,
                        view(replicas, :, x, k),
                        view(replicas, :, y, k),
                        view(replicas, :, z, k),
                        j, x_new, y_new, z_new)
                    if (delta <= _zero) || (rand(T) <= exp(-inv_temp * delta))
                        replicas[j, x, k] = x_new
                        replicas[j, y, k] = y_new
                        replicas[j, z, k] = z_new
                        energy += delta
                        num_accept += 1
                    else
                        num_reject += 1
                    end
                else
                    num_reject += 1
                end
                energies[i, k] = energy
            end
            if 3 * num_accept < num_reject
                perturbation_radii[k] = perturbation_radius / _fac
            elseif 3 * num_accept > num_reject
                perturbation_radii[k] = min(_one, perturbation_radius * _fac)
            end
        end
    end

    return nothing
end


function parallel_swap!(
    energy_function::F,
    replicas::AbstractArray{T,3},
    inverse_temperatures::AbstractVector{T},
    odd::Bool,
) where {F,T}

    _zero = zero(T)

    particle_axis = axes(replicas, 1)
    xyz_axis = axes(replicas, 2)
    replica_axis = axes(replicas, 3)
    @assert (particle_axis, xyz_axis, replica_axis) == axes(replicas)
    @assert (replica_axis,) == axes(inverse_temperatures)
    @assert length(xyz_axis) == 3
    x, y, z = xyz_axis

    _one_index = one(eltype(replica_axis))
    _two_index = twice(_one_index)
    first_index = odd ? first(replica_axis) + _one_index : first(replica_axis)
    last_index = last(replica_axis) - _one_index
    @inbounds begin
        @threads for k = first_index:_two_index:last_index
            a = k
            b = k + _one_index
            energy_a = pairwise_radial_energy(
                energy_function,
                view(replicas, :, x, a),
                view(replicas, :, y, a),
                view(replicas, :, z, a))
            energy_b = pairwise_radial_energy(
                energy_function,
                view(replicas, :, x, b),
                view(replicas, :, y, b),
                view(replicas, :, z, b))
            inv_temp_a = inverse_temperatures[a]
            inv_temp_b = inverse_temperatures[b]
            log_prob = (energy_a - energy_b) * (inv_temp_a - inv_temp_b)
            if (log_prob >= _zero) || (rand(T) <= exp(log_prob))
                temp = copy(view(replicas, :, :, a))
                copy!(view(replicas, :, :, a), view(replicas, :, :, b))
                copy!(view(replicas, :, :, b), temp)
            end
        end
    end

    return nothing
end


function analyze(
    energies::AbstractMatrix{T},
    inverse_temperatures::AbstractVector{T},
) where {T}

    _zero = zero(T)

    iteration_axis = axes(energies, 1)
    replica_axis = axes(energies, 2)
    @assert (iteration_axis, replica_axis) == axes(energies)
    @assert (replica_axis,) == axes(inverse_temperatures)

    cv = similar(Vector{T}, replica_axis)
    cv_prime = similar(Vector{T}, replica_axis)

    @inbounds for k in replica_axis
        V1 = _zero
        V2 = _zero
        V3 = _zero
        @simd for i in iteration_axis
            E = energies[i, k]
            E2 = square(E)
            V1 += E
            V2 += E2
            V3 += E2 * E
        end
        n = T(length(iteration_axis))
        V1 /= n
        V2 /= n
        V3 /= n
        inv_tau = inverse_temperatures[k]
        inv_tau_2 = square(inv_tau)
        inv_tau_4 = square(inv_tau_2)

        var = V2 - square(V1)
        cov = V3 - V2 * V1
        cv[k] = inv_tau_2 * var
        cv_prime[k] = inv_tau_4 * (cov - (V1 + inv(inv_tau)) * twice(var))
    end

    return cv, cv_prime
end


function main(;
    num_particles::Int,
    num_replicas::Int,
    min_temp::T,
    max_temp::T,
    constraining_radius::T,
    num_steps::Int,
    num_batches::Int,
) where {T}

    _zero = zero(T)
    _one = one(T)
    _two = twice(_one)
    _half = inv(_two)

    replicas = Array{T,3}(undef, num_particles, 3, num_replicas)
    for k = 1:num_replicas
        for i = 1:num_particles
            while true
                x = randn(T)
                y = randn(T)
                z = randn(T)
                if square(x) + square(y) + square(z) < square(constraining_radius)
                    replicas[i, 1, k] = x
                    replicas[i, 2, k] = y
                    replicas[i, 3, k] = z
                    break
                end
            end
        end
    end

    inv_temps = exp.(range(-log(min_temp), -log(max_temp), num_replicas))
    perturbation_radii = fill(_half^12, num_replicas)

    energies = Matrix{T}(undef, 2 * num_steps * num_batches, num_replicas)

    while true
        start = time_ns()
        for i = 1:num_batches
            even_view = view(energies, num_steps*(2*i-2)+1:num_steps*(2*i-1), :)
            odd_view = view(energies, num_steps*(2*i-1)+1:num_steps*(2*i-0), :)
            parallel_temper!(lj_energy, even_view, replicas, inv_temps,
                perturbation_radii, constraining_radius)
            parallel_swap!(lj_energy, replicas, inv_temps, false)
            parallel_temper!(lj_energy, odd_view, replicas, inv_temps,
                perturbation_radii, constraining_radius)
            parallel_swap!(lj_energy, replicas, inv_temps, true)
        end
        stop = time_ns()
        duration = (stop - start) / 1.0e9
        num_trials = 2 * num_steps * num_batches * num_replicas

        cv, cv_prime = analyze(energies, inv_temps)
        p1 = plot(inv.(inv_temps), perturbation_radii, label="R(T)", color=:green,
            xlimits=(min_temp, max_temp), ylimits=(_zero, _one))
        p2 = plot(inv.(inv_temps), cv, label="C_V(T)", color=:blue,
            xlimits=(min_temp, max_temp))
        p3 = plot(inv.(inv_temps), cv_prime, label="C_V'(T)", color=:red,
            xlimits=(min_temp, max_temp))
        display(plot(p1, p2, p3, layout=(1, 3)))
        println("Monte Carlo steps per second: ", num_trials / duration)
        flush(stdout)
    end

    return nothing
end


main(;
    num_particles=38,
    num_replicas=256,
    min_temp=Float64(0.05),
    max_temp=Float64(0.35),
    constraining_radius=Float64(2.25),
    num_steps=500,
    num_batches=1000,
)
