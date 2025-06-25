module ExampleFunctions


# TODO: twice(x) and square(x) should be moved into MultiFloats.jl v3.
@inline twice(x) = x + x
@inline square(x) = x * x


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


function pairwise_radial_energy(
    energy_function::F,
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T},
) where {F,T}

    _zero = zero(T)

    point_axis = axes(x, 1)
    @assert (point_axis,) == axes(x)
    @assert (point_axis,) == axes(y)
    @assert (point_axis,) == axes(z)

    energy = _zero
    @inbounds for i in point_axis
        xi = x[i]
        yi = y[i]
        zi = z[i]
        @simd for j = i+one(i):last(point_axis)
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

    point_axis = axes(gx, 1)
    @assert (point_axis,) == axes(gx)
    @assert (point_axis,) == axes(gy)
    @assert (point_axis,) == axes(gz)
    @assert (point_axis,) == axes(x)
    @assert (point_axis,) == axes(y)
    @assert (point_axis,) == axes(z)

    @inbounds for i in point_axis
        xi = x[i]
        yi = y[i]
        zi = z[i]
        ax = _zero
        ay = _zero
        az = _zero
        @simd for j in point_axis
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

    point_axis = axes(px, 1)
    @assert (point_axis,) == axes(px)
    @assert (point_axis,) == axes(py)
    @assert (point_axis,) == axes(pz)
    @assert (point_axis,) == axes(x)
    @assert (point_axis,) == axes(y)
    @assert (point_axis,) == axes(z)
    @assert (point_axis,) == axes(u)
    @assert (point_axis,) == axes(v)
    @assert (point_axis,) == axes(w)

    @inbounds for i in point_axis
        xi = x[i]
        yi = y[i]
        zi = z[i]
        ui = u[i]
        vi = v[i]
        wi = w[i]
        ax = _zero
        ay = _zero
        az = _zero
        @simd for j in point_axis
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


end # module ExampleFunctions
