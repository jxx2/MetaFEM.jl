function modify_Omega(v1, v2)
    angle = sqrt(2.) / 2
    v1_norm, v2_norm = (v1, v2) .|> norm
    v1_dot_v2 = dot(v1, v2)
    rho = abs(v1_dot_v2 / (v1_norm * v2_norm))
    omega = v1_dot_v2 / (v1_norm * v1_norm)
    return (rho < angle) ? (omega * angle / rho) : omega
end

"""
    idrs!(x, A, b; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 4, kwargs...)

IDRs (induced dimension reduction) solver, adapted from IterativeSolvers.jl.
"""
idrs!

"""
    idrs_original!(x, A, b; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 4, kwargs...)

IDRs (induced dimension reduction) solver, where the "original" suffix is the earlier version doing direct orthogonalization without exploit orthogonality incrementally.
"""
idrs_original!

for ArrayType in FEM_ArrayTypes
    @eval begin
        function idrs!(x::$ArrayType{Tb, 1}, A, b::$ArrayType{Tb, 1}; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 4, kwargs...) where {Tb}
            r = Pl(b - A * x)
            (normalized_norm(r) <= tol) && return 0
            iter = 1

            Lb = length(b)
            Ar = zero(b)
            P = $ArrayType{Tb, 1}[FEM_rand($ArrayType, Tb, Lb) for k in 1:s]
            U, G = $ArrayType{Tb, 1}[FEM_zeros($ArrayType, Tb, Lb) for k in 1:s], $ArrayType{Tb, 1}[FEM_zeros($ArrayType, Tb, Lb) for k in 1:s]
            Q, V = FEM_zeros($ArrayType, Tb, Lb), FEM_zeros($ArrayType, Tb, Lb)
            M, f, c = Matrix{Tb}(LinearAlgebra.I, s, s), zeros(Tb, s), zeros(Tb, s) #CPU matrix for Mc = f
            
            omega::Tb = 1.
            while true
                for i in 1:s
                    f[i] = dot(P[i], r)
                end
                for k in 1:s
                    # Solve small system and make v orthogonal to P
                    c = LowerTriangular(M[k:s, k:s]) \ f[k:s]

                    V .= c[1] .* G[k]
                    Q .= c[1] .* U[k]
                    for i = (k + 1):s
                        V .+= c[i - k + 1] .* G[i]
                        Q .+= c[i - k + 1] .* U[i]
                    end
                    # Compute new U[k] and G[k], G[k] is in space G_j
                    V .= r .- V
                    U[k] .= Q .+ omega .* V
                    mul!(G[k], A, U[k])
                    Pl(G[k])

                    # Bi-orthogonalise the new basis vectors, is serial
                    for i in 1:(k - 1)
                        alpha = dot(P[i], G[k]) / M[i, i]
                        G[k] .-= alpha .* G[i]
                        U[k] .-= alpha .* U[i]
                    end

                    # New column of M = P'*G  (first k-1 entries are zero)  
                    for i in k:s
                        M[i, k] = dot(P[i], G[k])
                    end

                    #  Make r orthogonal to q_i, i = 1..k
                    beta = f[k] / M[k, k]

                    x .+= beta .* U[k]
                    r .-= beta .* G[k]

                    (normalized_norm(r) <= tol || iter >= maxiter) && return iter
                    f[(k + 1):s] .-= beta * M[(k + 1):s, k]
                    iter += 1
                end
                # Now we have sufficient vectors in G_j to comegapute residual in G_j+1
                # r = b - A * x
                mul!(Ar, A, r)
                Pl(Ar)
                omega = modify_Omega(Ar, r) 

                x .+= omega .* r
                r .-= omega .* Ar

                (normalized_norm(r) <= tol || iter >= maxiter) && return iter
                iter += 1
            end
        end

        #not used, not exploiting orthogonality
        function idrs_original!(x::$ArrayType{Tb, 1}, A, b::$ArrayType{Tb, 1}; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 4, kwargs...) where {Tb}
            r = Pl(b - A * x)
            (normalized_norm(r) <= tol) && return 0
            iter = 1

            Lb = length(b)
            Ar = zero(b)
            P = $ArrayType{Tb, 1}[FEM_rand($ArrayType, Tb, Lb) for k in 1:s]
            U, G = $ArrayType{Tb, 1}[FEM_zeros($ArrayType, Tb, Lb) for k in 1:s], $ArrayType{Tb, 1}[FEM_zeros($ArrayType, Tb, Lb) for k in 1:s]
            Q, V = FEM_zeros($ArrayType, Tb, Lb), FEM_zeros($ArrayType, Tb, Lb)
            M, f, c = zeros(Tb, s, s), zeros(Tb, s), zeros(Tb, s) #CPU matrix for Mc = f
            
            omega::Tb = 1.
            for k in 1:s
                U[k] .= r
                mul!(G[k], A, r)
                Pl(G[k])

                omega = modify_Omega(G[k], r)
                x .+= omega .* U[k]
                r .-= omega .* G[k]
                for i = 1:s
                    M[i, k] = dot(P[i], G[k])
                end
            end

            while true
                # r = b - A * x
                (normalized_norm(r) <= tol || iter >= maxiter) && return iter
                iter += 1

                for k in 0:s
                    for i in 1:s
                        f[i] = dot(P[i], r)
                    end
                    c = M \ f

                    V .= c[1] .* G[1]
                    Q .= c[1] .* U[1]
                    for i = 2:s
                        V .+= c[i] .* G[i]
                        Q .+= c[i] .* U[i]
                    end

                    V .= r .- V
                    if k == 0
                        mul!(Ar, A, V)
                        Pl(Ar)
                        omega = modify_Omega(Ar, V) 

                        Q .+= omega .* V
                        x .+= Q 
                        mul!(r, A, Q)
                        Pl(r)
                    else
                        U[k] .= Q .+ omega .* V
                        mul!(G[k], A, U[k])
                        Pl(G[k])

                        x .+= U[k]
                        r .-= G[k]

                        for i in 1:s
                            M[i, k] = dot(P[i], G[k])
                        end
                    end 
                end
            end
        end
    end
end