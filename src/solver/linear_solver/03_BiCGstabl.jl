"""
    bicgstabl!(x, A, b, R; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 2, kwargs...)
    
bicgstabl, a classical stablized bi conjugate gradient method, adapted from IterativeSolvers.jl.
"""
bicgstabl!

"""
    bicgstabl_GS!(x, A, b, R; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 2, kwargs...)
    
bicgstabl, a classical stablized bi conjugate gradient method, where the "GS" suffix means using Gram-Schmidt instead of LU to solve the minimal residual part, 
empirically more numerical stable.
"""
bicgstabl_GS!

for ArrayType in FEM_ArrayTypes
    @eval begin
        function bicgstabl_GS!(x::$ArrayType{Tb, 1}, A, b::$ArrayType{Tb, 1}, r::$ArrayType{Tb, 1}; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 2, kwargs...) where {Tb}
            mul!(r, A, x, -1.)
            r .+= b
            Pl(r)

            r_norm = normalized_norm(r)
            (r_norm <= tol) && return 0
            iter = 1

            Lb = length(b)
            γ = zeros(Tb, s)
            γp = zeros(Tb, s)
            γpp = zeros(Tb, s)
            σ = zeros(Tb, s)
            tau = zeros(Tb, s, s)

            ω = ρ0 = one(Tb)
            α = zero(Tb)

            r_shadow = FEM_rand($ArrayType, Tb, Lb)
            R = [FEM_buffer($ArrayType, Tb, Lb) for i = 1:(s + 1)]
            R[1] = r
            U = [FEM_buffer($ArrayType, Tb, Lb) for i = 1:(s + 1)]
            while true
                ρ0 *= - ω
                # BiCG part
                for j = 1:s
                    ρ1 = dot(r_shadow, R[j])
                    β = α * ρ1 / ρ0
                    ρ0 = ρ1

                    for i = 1:j
                        U[i] .= R[i] .- β .* U[i]
                    end
                    mul!(U[j + 1], A, U[j])
                    Pl(U[j + 1])
                    α = ρ0 / dot(r_shadow, U[j + 1])

                    for i = 1:j
                        R[i] .-= α .* U[i + 1]
                    end
                    mul!(R[j + 1], A, R[j])
                    Pl(R[j + 1])
                    x .+= α .* U[1]
                end

                # MR part
                for j = 1:s
                    for i = 1:(j - 1)
                        tau[i, j] = dot(R[i + 1], R[j + 1]) / σ[i]
                        R[j + 1] .-= tau[i, j] .* R[i + 1]
                    end
                    σ[j] = dot(R[j + 1], R[j + 1])
                    γp[j] = dot(R[1], R[j + 1]) / σ[j]
                end
                γ[s] = γp[s]
                ω = γ[s]

                for j = (s - 1):(-1):1
                    γ[j] = γp[j] - dot(tau[j, (j + 1):s], γ[(j + 1):s])
                end
                for j = 1:(s - 1)
                    γpp[j] = γ[j + 1] + dot(tau[j, (j + 1):(s - 1)], γ[(j + 2):s])
                end

                x .+= γ[1] .* R[1]
                R[1] .-= γp[s] .* R[s + 1]
                U[1] .-=  γ[s] .* U[s + 1]

                for j = 1:(s - 1)
                    U[1] .-= γ[j] .* U[j + 1]
                    x .+= γpp[j] .* R[j + 1]
                    R[1] .-= γp[j] .* R[j + 1]
                end

                iter += s
                ((normalized_norm(R[1]) <= tol) || iter >= maxiter) && return iter
            end
        end

        #with lu, slightly unstable than the above one, it happened
        function bicgstabl!(x::$ArrayType{Tb, 1}, A, b::$ArrayType{Tb, 1}, r::$ArrayType{Tb, 1}; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 2, kwargs...) where {Tb}
            mul!(r, A, x, -1.)
            r .+= b
            Pl(r)
            r_norm = norm(r)
            (r_norm <= tol) && return 0
            iter = 1

            Lb = length(b)
            γ = zeros(Tb, s)
            ω = ρ0 = one(Tb)
            α = zero(Tb)

            r_shadow = FEM_rand($ArrayType, Tb, Lb)
            R = [FEM_buffer($ArrayType, Tb, Lb) for i = 1:(s + 1)]
            R[1] = r
            U = [FEM_buffer($ArrayType, Tb, Lb) for i = 1:(s + 1)]
            M = zeros(Tb, s + 1, s + 1)

            while true
                ρ0 *= - ω
                # BiCG part
                for j = 1:s
                    ρ1 = dot(r_shadow, R[j])
                    β = α * ρ1 / ρ0
                    ρ0 = ρ1

                    for i = 1:j
                        U[i] .= R[i] .- β .* U[i]
                    end
                    mul!(U[j + 1], A, U[j])
                    Pl(U[j + 1])
                    α = ρ0 / dot(r_shadow, U[j + 1])

                    for i = 1:j
                        R[i] .-= α .* U[i + 1]
                    end
                    mul!(R[j + 1], A, R[j])
                    Pl(R[j + 1])
                    x .+= α .* U[1]
                end

                # MR part
                L = 2:(s + 1)
                for j = 1:(s + 1)
                    M[j, j] = dot(R[j], R[j])
                    for i = 1:(j - 1)
                        M[i, j] = dot(R[i], R[j])
                        M[j, i] = M[i, j]
                    end
                end
                ldiv!(γ, lu!(view(M, L, L)), view(M, L, 1))

                for i = 1:s
                    U[1] .-= γ[i] .* U[i + 1]
                    x .+= γ[i] .* R[i]
                end
                for i = 1:s
                    R[1] .-= γ[i] .* R[i + 1]
                end

                ω = γ[s]
                iter += s
                ((normalized_norm(R[1]) <= tol) || iter >= maxiter) && return iter
            end
        end
    end
end