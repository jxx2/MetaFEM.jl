"""
    solver_BiCG(globalfield; Pl_func::Function = precondition_Nothing, max_iter = 5000, max_pass = 4, l = 8)

GPU bicgstabl (a classical stablized bi conjugate gradient method) solver.
"""
function solver_BiCG(globalfield; Pl_func::Function = precondition_Nothing, max_iter = 5000, max_pass = 4, l = 8)
    @Takeout (K_I, K_J, K_val_ids, K_total, residue, converge_tol) FROM globalfield

    K_vals = K_total[K_val_ids]
    A = CuSparseMatrixCSR(CuSparseMatrixCOO{FEM_Float}(K_I, K_J, K_vals))
    Pl = Pl_func(A)

    b = residue
    x = CUDA.zeros(FEM_Float, length(residue))
    println("BiCGstabl iterative solver with initial res = ", normalized_norm(b), " and target res = ", converge_tol)

    pass_number = 1
    tol_factor = 1.
    while true
        # pass_iters = bicgstabl_gpu!(x, A, b; Pl = Pl, tol = tol_factor * converge_tol, maxiter = max_iter,  l = l)
        pass_iters = bicgstabl_gpu_GS!(x, A, b; Pl = Pl, tol = tol_factor * converge_tol, maxiter = max_iter,  l = l)

        r = b - A * x
        res = normalized_norm(r)
        ldiv!(Pl, r)
        preconditioned_res = normalized_norm(r)
        tol_factor = min(preconditioned_res / res, 1.)

        if (res < converge_tol) || (pass_number >= max_pass)
            println("pass ", pass_number, " with res = ", res, " preconditioned res = ", preconditioned_res, ", iter = ", pass_iters, ", break")
            return x
        else
            println("pass ", pass_number, " with res = ", res, " preconditioned res = ", preconditioned_res, " iter = ", pass_iters)
            pass_number += 1
        end
    end
    x
end

function bicgstabl_gpu_GS!(x, A, b::CuVector{Tb}; Pl = Identity(), tol::Real, maxiter::Integer, l::Integer = 2) where Tb
    r0 = ldiv!(Pl, b - A * x)
    r_norm = normalized_norm(r0)
    iter = 1
    (r_norm <= tol) && return 0

    # Tb, Lb = Ref(b) .|> (eltype, length)
    Lb = length(b)
    γ = zeros(Tb, l)
    γp = zeros(Tb, l)
    γpp = zeros(Tb, l)
    σ = zeros(Tb, l)
    tau = zeros(Tb, l, l)

    ω = ρ0 = one(Tb)
    α = zero(Tb)

    r_shadow = CUDA.rand(Tb, Lb)
    r = CuVector{Tb}[CUDA.zeros(Tb, Lb) for i = 1:(l + 1)]
    r[1] = r0
    u = CuVector{Tb}[CUDA.zeros(Tb, Lb) for i = 1:(l + 1)]

    while true
        ρ0 *= - ω
        # BiCG part
        for j = 1:l
            ρ1 = dot(r_shadow, r[j])
            β = α * ρ1 / ρ0
            ρ0 = ρ1

            u[1:j] .= r[1:j] .- β .* u[1:j]
            mul!(u[j + 1], A, u[j])
            ldiv!(Pl, u[j + 1])
            α = ρ0 / dot(r_shadow, u[j + 1])

            r[1:j] .-= α .* u[2:(j + 1)]
            mul!(r[j + 1], A, r[j])
            ldiv!(Pl, r[j + 1])
            x .+= α * u[1]
        end

        # MR part
        for j = 1:l
            for i = 1:(j - 1)
                tau[i, j] = dot(r[i + 1], r[j + 1]) / σ[i]
                r[j + 1] -= tau[i, j] * r[i + 1]
            end
            σ[j] = dot(r[j + 1], r[j + 1])
            γp[j] = dot(r[1], r[j + 1]) / σ[j]
        end
        γ[l] = γp[l]
        ω = γ[l]

        for j = (l - 1):(-1):1
            γ[j] = γp[j] - dot(tau[j, (j + 1):l], γ[(j + 1):l])
        end
        for j = 1:(l - 1)
            γpp[j] = γ[j + 1] + dot(tau[j, (j + 1):(l - 1)], γ[(j + 2):l])
        end

        x .+= γ[1] * r[1]
        r[1] .-= γp[l] * r[l + 1]
        u[1] .-=  γ[l] * u[l + 1]

        for j = 1:(l - 1)
            u[1] .-= γ[j] * u[j + 1]
            x .+= γpp[j] * r[j + 1]
            r[1] .-= γp[j] * r[j + 1]
        end

        iter += l
        ((normalized_norm(r[1]) <= tol) || iter >= maxiter) && return iter
    end
end

#slightly unstable than the above one, it happened
function bicgstabl_gpu!(x, A, b::T; Pl = Identity(), tol::Real, maxiter::Integer, l::Integer = 2) where T
    r0 = ldiv!(Pl, b - A * x)
    r_norm = norm(r0)
    iter = 1
    (r_norm <= tol) && return x

    Tb, Lb = Ref(b) .|> (eltype, length)
    γ = zeros(Tb, l)
    ω = ρ0 = one(Tb)
    α = zero(Tb)

    r_shadow = CUDA.rand(Tb, Lb)
    r = T[CUDA.zeros(Tb, Lb) for i = 1:(l + 1)]
    r[1] = r0
    u = T[CUDA.zeros(Tb, Lb) for i = 1:(l + 1)]
    M = zeros(Tb, l + 1, l + 1)

    while true
        ρ0 *= - ω
        # BiCG part
        for j = 1:l
            ρ1 = dot(r_shadow, r[j])
            β = α * ρ1 / ρ0
            ρ0 = ρ1

            u[1:j] .= r[1:j] .- β .* u[1:j]
            mul!(u[j + 1], A, u[j])
            ldiv!(Pl, u[j + 1])
            α = ρ0 / dot(r_shadow, u[j + 1])

            r[1:j] .-= α .* u[2:(j + 1)]
            mul!(r[j + 1], A, r[j])
            ldiv!(Pl, r[j + 1])
            x .+= α * u[1]
        end

        # MR part
        L = 2:(l + 1)
        for j = 1:(l + 1)
            M[j, j] = dot(r[j], r[j])
            for i = 1:(j - 1)
                M[i, j] = dot(r[i], r[j])
                M[j, i] = M[i, j]
            end
        end
        ldiv!(γ, lu!(view(M, L, L)), view(M, L, 1))

        for i = 1:l
            mul!(u[1], u[i + 1], γ[i], -one(Tb), one(Tb))
            mul!(x, r[i], γ[i], one(Tb), one(Tb))
        end
        for i = 1:l
            mul!(r[1], r[i + 1], γ[i], -one(Tb), one(Tb))
        end

        ω = γ[l]
        iter += l
        ((normalized_norm(r[1]) <= tol) || iter >= maxiter) && return iter
    end
end

