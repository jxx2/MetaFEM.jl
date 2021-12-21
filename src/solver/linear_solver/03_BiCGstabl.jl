"""
    bicgstabl!(x, A, b; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 2, kwargs...)
    
bicgstabl, a classical stablized bi conjugate gradient method, adapted from IterativeSolvers.jl.
"""
bicgstabl!

"""
    bicgstabl_GS!(x, A, b; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 2, kwargs...)
    
bicgstabl, a classical stablized bi conjugate gradient method, where the "GS" suffix means using Gram-Schmidt instead of LU to solve the minimal residual part, 
empirically more numerical stable.
"""
bicgstabl_GS!

for ArrayType in FEM_ArrayTypes
    @eval begin
        function bicgstabl_GS!(x::$ArrayType{Tb, 1}, A, b::$ArrayType{Tb, 1}; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 2, kwargs...) where {Tb}
            r0 = Pl(b - A * x)
            r_norm = normalized_norm(r0)
            (r_norm <= tol) && return 0
            iter = 1
            # Tb, Lb = Ref(b) .|> (eltype, length)
            Lb = length(b)
            γ = zeros(Tb, s)
            γp = zeros(Tb, s)
            γpp = zeros(Tb, s)
            σ = zeros(Tb, s)
            tau = zeros(Tb, s, s)

            ω = ρ0 = one(Tb)
            α = zero(Tb)

            r_shadow = FEM_rand($ArrayType, Tb, Lb)
            r = $ArrayType{Tb, 1}[FEM_zeros($ArrayType, Tb, Lb) for i = 1:(s + 1)]
            r[1] = r0
            u = $ArrayType{Tb, 1}[FEM_zeros($ArrayType, Tb, Lb) for i = 1:(s + 1)]

            while true
                ρ0 *= - ω
                # BiCG part
                for j = 1:s
                    ρ1 = dot(r_shadow, r[j])
                    β = α * ρ1 / ρ0
                    ρ0 = ρ1

                    u[1:j] .= r[1:j] .- β .* u[1:j]
                    mul!(u[j + 1], A, u[j])
                    Pl(u[j + 1])
                    α = ρ0 / dot(r_shadow, u[j + 1])

                    r[1:j] .-= α .* u[2:(j + 1)]
                    mul!(r[j + 1], A, r[j])
                    Pl(r[j + 1])
                    x .+= α * u[1]
                end

                # MR part
                for j = 1:s
                    for i = 1:(j - 1)
                        tau[i, j] = dot(r[i + 1], r[j + 1]) / σ[i]
                        r[j + 1] -= tau[i, j] * r[i + 1]
                    end
                    σ[j] = dot(r[j + 1], r[j + 1])
                    γp[j] = dot(r[1], r[j + 1]) / σ[j]
                end
                γ[s] = γp[s]
                ω = γ[s]

                for j = (s - 1):(-1):1
                    γ[j] = γp[j] - dot(tau[j, (j + 1):s], γ[(j + 1):s])
                end
                for j = 1:(s - 1)
                    γpp[j] = γ[j + 1] + dot(tau[j, (j + 1):(s - 1)], γ[(j + 2):s])
                end

                x .+= γ[1] * r[1]
                r[1] .-= γp[s] * r[s + 1]
                u[1] .-=  γ[s] * u[s + 1]

                for j = 1:(s - 1)
                    u[1] .-= γ[j] * u[j + 1]
                    x .+= γpp[j] * r[j + 1]
                    r[1] .-= γp[j] * r[j + 1]
                end

                iter += s
                ((normalized_norm(r[1]) <= tol) || iter >= maxiter) && return iter
            end
        end

        #with lu, slightly unstable than the above one, it happened
        function bicgstabl!(x::$ArrayType{Tb, 1}, A, b::$ArrayType{Tb, 1}; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 2, kwargs...) where {Tb}
            r0 = Pl(b - A * x)
            r_norm = norm(r0)
            (r_norm <= tol) && return 0
            iter = 1

            Lb = length(b)
            γ = zeros(Tb, s)
            ω = ρ0 = one(Tb)
            α = zero(Tb)

            r_shadow = FEM_rand($ArrayType, Tb, Lb)
            r = $ArrayType{Tb, 1}[FEM_zeros($ArrayType, Tb, Lb) for i = 1:(s + 1)]
            r[1] = r0
            u = $ArrayType{Tb, 1}[FEM_zeros($ArrayType, Tb, Lb) for i = 1:(s + 1)]
            M = zeros(Tb, s + 1, s + 1)

            while true
                ρ0 *= - ω
                # BiCG part
                for j = 1:s
                    ρ1 = dot(r_shadow, r[j])
                    β = α * ρ1 / ρ0
                    ρ0 = ρ1

                    u[1:j] .= r[1:j] .- β .* u[1:j]
                    mul!(u[j + 1], A, u[j])
                    Pl(u[j + 1])
                    α = ρ0 / dot(r_shadow, u[j + 1])

                    r[1:j] .-= α .* u[2:(j + 1)]
                    mul!(r[j + 1], A, r[j])
                    Pl(r[j + 1])
                    x .+= α * u[1]
                end

                # MR part
                L = 2:(s + 1)
                for j = 1:(s + 1)
                    M[j, j] = dot(r[j], r[j])
                    for i = 1:(j - 1)
                        M[i, j] = dot(r[i], r[j])
                        M[j, i] = M[i, j]
                    end
                end
                ldiv!(γ, lu!(view(M, L, L)), view(M, L, 1))

                for i = 1:s
                    mul!(u[1], u[i + 1], γ[i], -one(Tb), one(Tb))
                    mul!(x, r[i], γ[i], one(Tb), one(Tb))
                end
                for i = 1:s
                    mul!(r[1], r[i + 1], γ[i], -one(Tb), one(Tb))
                end

                ω = γ[s]
                iter += s
                ((normalized_norm(r[1]) <= tol) || iter >= maxiter) && return iter
            end
        end
    end
end