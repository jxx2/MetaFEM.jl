function solver_IDRs(globalfield; Pl_func::Function = precondition_Nothing, max_iter = 5000, max_pass = 4, s = 8)
    @Takeout (K_I, K_J, K_val_ids, K_total, residue, converge_tol) FROM globalfield

    K_vals = K_total[K_val_ids]
    A = CuSparseMatrixCSR(CuSparseMatrixCOO{FEM_Float}(K_I, K_J, K_vals))
    Pl = Pl_func(A)

    b = residue
    x = CUDA.zeros(FEM_Float, length(residue))

    println("IDRs iterative solver start with initial res = ", normalized_norm(b), " and target res = ", converge_tol)

    pass_number = 1
    tol_factor = 1.
    while true
        pass_iters = idrs_gpu!(x, A, b; Pl = Pl, tol = tol_factor * converge_tol, maxiter = max_iter,  s = s)

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

@inline function modify_Omega(v1, v2)
    angle = sqrt(2.) / 2
    v1_norm, v2_norm = (v1, v2) .|> norm
    v1_dot_v2 = dot(v1, v2)
    rho = abs(v1_dot_v2 / (v1_norm * v2_norm))
    omega = v1_dot_v2 / (v1_norm * v1_norm)
    return (rho < angle) ? (omega * angle / rho) : omega
end

function idrs_gpu!(x, A, b::T; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 4) where T
    r = ldiv!(Pl, b - A * x)
    iter = 1
    (normalized_norm(r) <= tol) && return 0

    Tb, Lb = Ref(b) .|> (eltype, length)
    Ar = zero(b)
    P = T[CUDA.rand(Tb, Lb) for k in 1:s]
    U, G = T[CUDA.zeros(Tb, Lb) for k in 1:s], T[CUDA.zeros(Tb, Lb) for k in 1:s]
    Q, V = CUDA.zeros(Tb, Lb), CUDA.zeros(Tb, Lb)
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
            ldiv!(Pl, G[k])

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
        ldiv!(Pl, Ar)
        omega = modify_Omega(Ar, r) 

        x .+= omega .* r
        r .-= omega .* Ar

        (normalized_norm(r) <= tol || iter >= maxiter) && return iter
        iter += 1
    end
end

#not used, not exploiting orthogonality
function idrs_gpu_original!(x, A, b::T; Pl = Identity(), tol::Real, maxiter::Integer, s::Integer = 4) where T
    r = ldiv!(Pl, b - A * x)
    iter = 1
    (normalized_norm(r) <= tol) && return 0

    Tb, Lb = Ref(b) .|> (eltype, length)
    Ar = zero(b)
    P = T[CUDA.rand(Tb, Lb) for k in 1:s]
    U, G = T[CUDA.zeros(Tb, Lb) for k in 1:s], T[CUDA.zeros(Tb, Lb) for k in 1:s]
    Q, V = CUDA.zeros(Tb, Lb), CUDA.zeros(Tb, Lb)
    M, f, c = zeros(Tb, s, s), zeros(Tb, s), zeros(Tb, s) #CPU matrix for Mc = f
    
    omega::Tb = 1.
    for k in 1:s
        U[k] .= r
        mul!(G[k], A, r)
        ldiv!(Pl, G[k])

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
                ldiv!(Pl, Ar)
                omega = modify_Omega(Ar, V) 

                Q .+= omega .* V
                x .+= Q 
                r .-= ldiv!(Pl, A * Q)
            else
                U[k] .= Q .+ omega .* V
                mul!(G[k], A, U[k])
                ldiv!(Pl, G[k])

                x .+= U[k]
                r .-= G[k]

                for i in 1:s
                    M[i, k] = dot(P[i], G[k])
                end
            end 
        end
    end
end
