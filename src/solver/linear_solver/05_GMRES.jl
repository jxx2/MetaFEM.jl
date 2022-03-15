import LinearAlgebra: Givens, givensAlgorithm
"""
    Hessenberg(H::AbstractMatrix, rhs::AbstractVector)

Solves `Hy = rhs` where H is `(n + 1) × n` matrix, rhs is a n + 1 vector and y is stored in the first n elements of rhs, the same as IterativeSolvers.jl/src/hessenberg.jlᵀ
"""
function Hessenberg(H::AbstractMatrix, rhs::AbstractVector)
    # Implicitly computes H = QR via Given's rotations
    # and then computes the least-squares solution y to
    # |Hy - rhs| = |QRy - rhs| = |Ry - Q'rhs|
    width = size(H, 2)

    # Hessenberg -> UpperTriangular; also apply to r.h.s.
    @inbounds for i = 1 : width
        c, s, _ = givensAlgorithm(H[i, i], H[i + 1, i])

        # Skip the first sub-diagonal since it'll be zero by design.
        H[i, i] = c * H[i, i] + s * H[i + 1, i]

        # Remaining columns
        @inbounds for j = i + 1 : width
            tmp = -conj(s) * H[i, j] + c * H[i + 1, j]
            H[i, j] = c * H[i, j] + s * H[i + 1, j]
            H[i + 1, j] = tmp
        end

        # Right hand side
        tmp = -conj(s) * rhs[i] + c * rhs[i + 1]
        rhs[i] = c * rhs[i] + s * rhs[i + 1]
        rhs[i + 1] = tmp
    end

    # Solve the upper triangular problem.
    U = UpperTriangular(view(H, 1 : width, 1 : width))
    ldiv!(U, view(rhs, 1 : width))
    nothing
end

"""
    gmres!(x, A, b, r; Pl = Identity(), tol::Real, maxiter::Integer, s = 20, kwargs...)

Vanilla GMRES solver which restarts every s = 20 iteration. GMRES may enter early stagnation and leave with large residue/error.
"""
gmres!

for ArrayType in FEM_ArrayTypes
    @eval begin
        function gmres!(x::$ArrayType{Tb, 1}, A, b::$ArrayType{Tb, 1}, r::$ArrayType{Tb, 1}; Pl = Identity(), tol::Real, maxiter::Integer, s = 20, kwargs...) where {Tb}
            mul!(r, A, x, -1.)
            r .+= b
            Pl(r)
            (normalized_norm(r) <= tol) && return 0
            iter = 1
            Lb = length(b)
            Q = [FEM_buffer($ArrayType, Tb, Lb) for k in 1:(s + 1)]
            H, y = Matrix{Tb}(undef, s + 1, s), zeros(Tb, s + 1)
            r_norm = norm(r)
            y[1] = r_norm
            while true
                Q[1] .= r ./ r_norm  
                for i = 2:(s + 1)
                    mul!(Q[i], A, Q[i - 1]) #now Q[i] is A * Q[i - 1]
                    Pl(Q[i])

                    for j = 1:(i - 1)
                        H[j, i - 1] = dot(Q[j], Q[i])
                        Q[i] .-= H[j, i - 1] .* Q[j]
                    end 

                    H[i, i - 1] = norm(Q[i])

                    if H[i, i - 1] == 0 # Note if Q[i] is zero then the problem is exactly solved
                        Hessenberg(H[1:(i - 1), 1:(i - 2)], y)
                        for j = 1:(i - 1)
                            x .+= (Q[j] .* y[j])
                        end
                        iter += (i - 1)
                        return iter
                    end

                    Q[i] ./= H[i, i - 1] # note Q[s + 1] does actually not need to be normalized
                end

                Hessenberg(H, y)
                for i = 1:s
                    x .+= (Q[i] .* y[i])
                end

                iter += s
                mul!(r, A, x, -1)
                r .+= b
                Pl(r) # r = Pl(b - A * x)

                ((normalized_norm(r) <= tol) || (iter > maxiter)) && return iter

                y .= 0
                r_norm = norm(r)
                y[1] = r_norm
            end
        end
    end
end