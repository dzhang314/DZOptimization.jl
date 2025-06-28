module ExampleFunctions

using KernelAbstractions: allocate, get_backend,
    @kernel, @Const, @index, @uniform

# TODO: twice(x) and square(x) should be moved into MultiFloats.jl v3.
@inline twice(x) = x + x
@inline square(x) = x * x

######################################################## RADIAL ENERGY FUNCTIONS


export lj_energy, lj_first_derivative, lj_second_derivative


@inline function lj_energy(r2::T) where {T}

    _one = one(T)
    _two = twice(_one)
    _four = twice(_two)

    inv_r2 = inv(r2)
    inv_r4 = square(inv_r2)
    inv_r6 = inv_r4 * inv_r2

    return _four * fma(inv_r6, inv_r6, -inv_r6)
end


@inline function lj_first_derivative(r2::T) where {T}

    _one = one(T)
    _two = twice(_one)
    _four = twice(_two)
    _eight = twice(_four)
    _twelve = _eight + _four
    _neg_twelve = -_twelve

    inv_r2 = inv(r2)
    inv_r4 = square(inv_r2)
    inv_r6 = inv_r4 * inv_r2
    inv_r8 = square(inv_r4)

    # TODO: Benchmark twice(inv_r8) instead of twice(inv_r6).
    # Accuracy should be identical, but there may be a performance benefit.
    return _neg_twelve * fma(inv_r8, twice(inv_r6), -inv_r8)
end


@inline function lj_second_derivative(r2::T) where {T}

    _one = one(T)
    _two = twice(_one)
    _three = _two + _one
    _four = twice(_two)
    _seven = _four + _three
    _eight = twice(_four)
    _sixteen = twice(_eight)
    _thirty_two = twice(_sixteen)
    _forty_eight = _thirty_two + _sixteen
    _seven_halves = _seven / _two

    inv_r2 = inv(r2)
    inv_r4 = square(inv_r2)
    inv_r8 = square(inv_r4)
    inv_r10 = inv_r8 * inv_r2

    # TODO: Benchmark variations of this formula.
    # The following variant may improve accuracy.
    # return _forty_eight * fma(_seven_halves * inv_r8, inv_r8, -inv_r10)
    return _forty_eight * fma(_seven_halves, square(inv_r8), -inv_r10)
end


######################################################## N-BODY ENERGY FUNCTIONS


export pairwise_radial_energy, accelerated_pairwise_radial_energy,
    pairwise_radial_gradient!, accelerated_pairwise_radial_gradient!,
    pairwise_radial_hvp!, accelerated_pairwise_radial_hvp!


function pairwise_radial_energy(
    energy_function::F,
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T},
) where {F,T}

    _zero = zero(T)

    particle_axis = axes(x, 1)
    @assert (particle_axis,) == axes(x)
    @assert (particle_axis,) == axes(y)
    @assert (particle_axis,) == axes(z)

    energy = _zero
    @inbounds for i in particle_axis
        xi = x[i]
        yi = y[i]
        zi = z[i]
        @simd for j = i+one(i):last(particle_axis)
            xj = x[j]
            yj = y[j]
            zj = z[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            r2 = square(dx) + square(dy) + square(dz)
            energy += energy_function(r2)
        end
    end
    return energy
end


@kernel function pairwise_radial_point_energy_kernel!(
    point_energies::AbstractVector{T},
    n::Int,
    energy_function::F,
    @Const(x::AbstractVector{T}),
    @Const(y::AbstractVector{T}),
    @Const(z::AbstractVector{T}),
) where {F,T}

    @uniform _zero = zero(T)
    @uniform _one = one(T)
    @uniform _two = twice(_one)
    @uniform _half = inv(_two)

    i = @index(Global, Linear)
    @inbounds begin
        xi = x[i]
        yi = y[i]
        zi = z[i]
        energy = _zero
        for j = 1:n
            xj = x[j]
            yj = y[j]
            zj = z[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            r2 = square(dx) + square(dy) + square(dz)
            energy += ifelse(i == j, _zero, energy_function(r2))
        end
        point_energies[i] = _half * energy
    end
end


function accelerated_pairwise_radial_energy(
    energy_function::F,
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T};
    workgroupsize::Int=256,
) where {F,T}

    n = size(x, 1)
    @assert (Base.OneTo(n),) == axes(x)
    @assert (Base.OneTo(n),) == axes(y)
    @assert (Base.OneTo(n),) == axes(z)

    backend = get_backend(x)
    @assert backend == get_backend(y)
    @assert backend == get_backend(z)

    kernel = pairwise_radial_point_energy_kernel!(backend, workgroupsize)
    point_energies = allocate(backend, T, n)
    kernel(point_energies, n, energy_function, x, y, z; ndrange=n)
    return sum(point_energies)
end


function pairwise_radial_gradient!(
    gx::AbstractVector{T},
    gy::AbstractVector{T},
    gz::AbstractVector{T},
    energy_first_derivative::F,
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T},
) where {F,T}

    _zero = zero(T)

    particle_axis = axes(gx, 1)
    @assert (particle_axis,) == axes(gx)
    @assert (particle_axis,) == axes(gy)
    @assert (particle_axis,) == axes(gz)
    @assert (particle_axis,) == axes(x)
    @assert (particle_axis,) == axes(y)
    @assert (particle_axis,) == axes(z)

    @inbounds for i in particle_axis
        xi = x[i]
        yi = y[i]
        zi = z[i]
        ax = _zero
        ay = _zero
        az = _zero
        @simd for j in particle_axis
            xj = x[j]
            yj = y[j]
            zj = z[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            r2 = square(dx) + square(dy) + square(dz)
            f = ifelse(i == j, _zero, energy_first_derivative(r2))
            ax += f * dx
            ay += f * dy
            az += f * dz
        end
        gx[i] = twice(ax)
        gy[i] = twice(ay)
        gz[i] = twice(az)
    end
    return nothing
end


@kernel function pairwise_radial_gradient_kernel!(
    gx::AbstractVector{T},
    gy::AbstractVector{T},
    gz::AbstractVector{T},
    n::Int,
    energy_first_derivative::F,
    @Const(x::AbstractVector{T}),
    @Const(y::AbstractVector{T}),
    @Const(z::AbstractVector{T}),
) where {F,T}

    @uniform _zero = zero(T)

    i = @index(Global, Linear)
    @inbounds begin
        xi = x[i]
        yi = y[i]
        zi = z[i]
        ax = _zero
        ay = _zero
        az = _zero
        for j = 1:n
            xj = x[j]
            yj = y[j]
            zj = z[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            r2 = square(dx) + square(dy) + square(dz)
            f = ifelse(i == j, _zero, energy_first_derivative(r2))
            ax += f * dx
            ay += f * dy
            az += f * dz
        end
        gx[i] = twice(ax)
        gy[i] = twice(ay)
        gz[i] = twice(az)
    end
end


function accelerated_pairwise_radial_gradient!(
    gx::AbstractVector{T},
    gy::AbstractVector{T},
    gz::AbstractVector{T},
    energy_first_derivative::F,
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T};
    workgroupsize::Int=256,
) where {F,T}

    n = size(gx, 1)
    @assert (Base.OneTo(n),) == axes(gx)
    @assert (Base.OneTo(n),) == axes(gy)
    @assert (Base.OneTo(n),) == axes(gz)
    @assert (Base.OneTo(n),) == axes(x)
    @assert (Base.OneTo(n),) == axes(y)
    @assert (Base.OneTo(n),) == axes(z)

    backend = get_backend(gx)
    @assert backend == get_backend(gy)
    @assert backend == get_backend(gz)
    @assert backend == get_backend(x)
    @assert backend == get_backend(y)
    @assert backend == get_backend(z)

    kernel = pairwise_radial_gradient_kernel!(backend, workgroupsize)
    kernel(gx, gy, gz, n, energy_first_derivative, x, y, z; ndrange=n)
    return nothing
end


function pairwise_radial_hvp!(
    px::AbstractVector{T},
    py::AbstractVector{T},
    pz::AbstractVector{T},
    energy_first_derivative::F1,
    energy_second_derivative::F2,
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T},
    u::AbstractVector{T},
    v::AbstractVector{T},
    w::AbstractVector{T},
) where {F1,F2,T}

    _zero = zero(T)

    particle_axis = axes(px, 1)
    @assert (particle_axis,) == axes(px)
    @assert (particle_axis,) == axes(py)
    @assert (particle_axis,) == axes(pz)
    @assert (particle_axis,) == axes(x)
    @assert (particle_axis,) == axes(y)
    @assert (particle_axis,) == axes(z)
    @assert (particle_axis,) == axes(u)
    @assert (particle_axis,) == axes(v)
    @assert (particle_axis,) == axes(w)

    @inbounds for i in particle_axis
        xi = x[i]
        yi = y[i]
        zi = z[i]
        ui = u[i]
        vi = v[i]
        wi = w[i]
        ax = _zero
        ay = _zero
        az = _zero
        @simd for j in particle_axis
            xj = x[j]
            yj = y[j]
            zj = z[j]
            uj = u[j]
            vj = v[j]
            wj = w[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            du = ui - uj
            dv = vi - vj
            dw = wi - wj
            r2 = square(dx) + square(dy) + square(dz)
            f = ifelse(i == j, _zero, energy_first_derivative(r2))
            s = ifelse(i == j, _zero, energy_second_derivative(r2))
            overlap = dx * du + dy * dv + dz * dw
            # TODO: Benchmark variations of this formula.
            g = twice(overlap * s)
            # g = twice(overlap) * s
            # g = overlap * twice(s)
            ax += f * du + g * dx
            ay += f * dv + g * dy
            az += f * dw + g * dz
        end
        px[i] = twice(ax)
        py[i] = twice(ay)
        pz[i] = twice(az)
    end
    return nothing
end


@kernel function pairwise_radial_hvp_kernel!(
    px::AbstractVector{T},
    py::AbstractVector{T},
    pz::AbstractVector{T},
    n::Int,
    energy_first_derivative::F1,
    energy_second_derivative::F2,
    @Const(x::AbstractVector{T}),
    @Const(y::AbstractVector{T}),
    @Const(z::AbstractVector{T}),
    @Const(u::AbstractVector{T}),
    @Const(v::AbstractVector{T}),
    @Const(w::AbstractVector{T}),
) where {F1,F2,T}

    @uniform _zero = zero(T)

    i = @index(Global, Linear)
    @inbounds begin
        xi = x[i]
        yi = y[i]
        zi = z[i]
        ui = u[i]
        vi = v[i]
        wi = w[i]
        ax = _zero
        ay = _zero
        az = _zero
        for j = 1:n
            xj = x[j]
            yj = y[j]
            zj = z[j]
            uj = u[j]
            vj = v[j]
            wj = w[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            du = ui - uj
            dv = vi - vj
            dw = wi - wj
            r2 = square(dx) + square(dy) + square(dz)
            f = ifelse(i == j, _zero, energy_first_derivative(r2))
            s = ifelse(i == j, _zero, energy_second_derivative(r2))
            overlap = dx * du + dy * dv + dz * dw
            # TODO: Benchmark variations of this formula.
            g = twice(overlap * s)
            # g = twice(overlap) * s
            # g = overlap * twice(s)
            ax += f * du + g * dx
            ay += f * dv + g * dy
            az += f * dw + g * dz
        end
        px[i] = twice(ax)
        py[i] = twice(ay)
        pz[i] = twice(az)
    end
end


function accelerated_pairwise_radial_hvp!(
    px::AbstractVector{T},
    py::AbstractVector{T},
    pz::AbstractVector{T},
    energy_first_derivative::F1,
    energy_second_derivative::F2,
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T},
    u::AbstractVector{T},
    v::AbstractVector{T},
    w::AbstractVector{T};
    workgroupsize::Int=256,
) where {F1,F2,T}

    n = size(px, 1)
    @assert (Base.OneTo(n),) == axes(px)
    @assert (Base.OneTo(n),) == axes(py)
    @assert (Base.OneTo(n),) == axes(pz)
    @assert (Base.OneTo(n),) == axes(x)
    @assert (Base.OneTo(n),) == axes(y)
    @assert (Base.OneTo(n),) == axes(z)
    @assert (Base.OneTo(n),) == axes(u)
    @assert (Base.OneTo(n),) == axes(v)
    @assert (Base.OneTo(n),) == axes(w)

    backend = get_backend(px)
    @assert backend == get_backend(py)
    @assert backend == get_backend(pz)
    @assert backend == get_backend(x)
    @assert backend == get_backend(y)
    @assert backend == get_backend(z)
    @assert backend == get_backend(u)
    @assert backend == get_backend(v)
    @assert backend == get_backend(w)

    kernel = pairwise_radial_hvp_kernel!(backend, workgroupsize)
    kernel(px, py, pz, n,
        energy_first_derivative, energy_second_derivative,
        x, y, z, u, v, w; ndrange=n)
    return nothing
end


################################################################################


export pairwise_radial_energy_delta


function pairwise_radial_energy_delta(
    energy_function::F,
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T},
    i::I,
    x_new::T,
    y_new::T,
    z_new::T,
) where {F,T,I}

    _zero = zero(T)

    particle_axis = axes(x, 1)
    @assert (particle_axis,) == axes(x)
    @assert (particle_axis,) == axes(y)
    @assert (particle_axis,) == axes(z)
    @assert i in particle_axis

    energy_old = _zero
    energy_new = _zero
    @inbounds begin
        x_old = x[i]
        y_old = y[i]
        z_old = z[i]
        @simd for j in first(particle_axis):i-one(i)
            xj = x[j]
            yj = y[j]
            zj = z[j]
            dx_old = x_old - xj
            dy_old = y_old - yj
            dz_old = z_old - zj
            r2_old = square(dx_old) + square(dy_old) + square(dz_old)
            energy_old += energy_function(r2_old)
            dx_new = x_new - xj
            dy_new = y_new - yj
            dz_new = z_new - zj
            r2_new = square(dx_new) + square(dy_new) + square(dz_new)
            energy_new += energy_function(r2_new)
        end
        @simd for j in i+one(i):last(particle_axis)
            xj = x[j]
            yj = y[j]
            zj = z[j]
            dx_old = x_old - xj
            dy_old = y_old - yj
            dz_old = z_old - zj
            r2_old = square(dx_old) + square(dy_old) + square(dz_old)
            energy_old += energy_function(r2_old)
            dx_new = x_new - xj
            dy_new = y_new - yj
            dz_new = z_new - zj
            r2_new = square(dx_new) + square(dy_new) + square(dz_new)
            energy_new += energy_function(r2_new)
        end
    end
    return energy_new - energy_old
end


################################################################################

end # module ExampleFunctions
