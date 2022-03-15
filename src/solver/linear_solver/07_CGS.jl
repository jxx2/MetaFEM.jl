"""
    cgs!(x, A, b, r; Pl = Identity(), tol::Real, maxiter::Integer, kwargs...)
    cgs2!(x, A, b, r; Pl = Identity(), tol::Real, maxiter::Integer, kwargs...)
    tfqmr!(x, A, b, r; Pl = Identity(), tol::Real, maxiter::Integer, checkiter = 200, kwargs...)
    
CGS/CGS2/TFQMR can have strong oscillation before convergence. For some problems (like incompressible) they can converge significantly faster than BiCG type solvers.
For stability, CGS < CGS2 < TFQMR, while for efficiency CGS > CGS2 > TFQMR.
"""
cgs!

for ArrayType in FEM_ArrayTypes
    @eval begin
        function cgs!(x::$ArrayType{Tb, 1}, A, b::$ArrayType{Tb, 1}, r::$ArrayType{Tb, 1}; Pl = Identity(), tol::Real, maxiter::Integer, kwargs...) where {Tb}
            mul!(r, A, x, -1.)
            r .+= b
            Pl(r)
            (normalized_norm(r) <= tol) && return 0
            
            iter = 1    

            r0 = copy(r)
            Lb = length(r0)

            rho = rhobar = alpha = beta = one(Tb)
            u = FEM_buffer($ArrayType, Tb, Lb)
            p = FEM_buffer($ArrayType, Tb, Lb)
            s = FEM_buffer($ArrayType, Tb, Lb)
            v = FEM_buffer($ArrayType, Tb, Lb)

            while true
                rhobar = rho
                rho = dot(r, r0)
                beta = rho / rhobar

                s .= r .+ beta .* p
                u .= s .+ beta .* (p .+ beta .* u)
                mul!(v, A, u)
                Pl(v)

                alpha = rho / dot(v, r0)

                p .= s .- alpha .* v
                x .+= alpha .* (p .+ s)

                mul!(r, A, x, -1)
                r .+= b
                Pl(r)

                iter += 1
                ((normalized_norm(r) <= tol) || (iter > maxiter)) && return iter
            end 
        end

        function cgs2!(x::$ArrayType{Tb, 1}, A, b::$ArrayType{Tb, 1}, r::$ArrayType{Tb, 1}; Pl = Identity(), tol::Real, maxiter::Integer, kwargs...) where {Tb}
            mul!(r, A, x, -1.)
            r .+= b
            Pl(r)
            (normalized_norm(r) <= tol) && return 0
            
            iter = 1    

            r0 = copy(r)
            Lb = length(r0)

            s0 = FEM_rand($ArrayType, Tb, Lb)

            alpha = alphabar = beta = betabar = rho = rhobar = sigma = sigmabar = one(Tb)
            u = FEM_buffer($ArrayType, Tb, Lb)
            w = FEM_buffer($ArrayType, Tb, Lb)
            s = FEM_buffer($ArrayType, Tb, Lb)
            v = FEM_buffer($ArrayType, Tb, Lb)
            t = FEM_buffer($ArrayType, Tb, Lb)
            c = FEM_buffer($ArrayType, Tb, Lb)

            while true
                rho = dot(r, r0)
                beta = 1 / alphabar * rho / sigma

                v .= r .+ beta .* u
                rhobar = dot(r, s0)

                betabar = 1 / alpha * rhobar / sigmabar
                t .= r .+ betabar .* s 
                w .= t .+ beta .* (u .+ betabar .* w)

                mul!(c, A, w)
                Pl(c)
                sigma = dot(c, r0)
                alpha = rho / sigma
                s .= t .- alpha .* c

                sigmabar = dot(c, s0)
                alphabar = rhobar / sigmabar
                u .= v .- alphabar .* c

                x .+= alpha .* v .+ alphabar .* s

                mul!(r, A, x, -1)
                r .+= b
                Pl(r)

                iter += 1
                ((normalized_norm(r) <= tol) || (iter > maxiter)) && return iter
            end 
        end
    end
end

