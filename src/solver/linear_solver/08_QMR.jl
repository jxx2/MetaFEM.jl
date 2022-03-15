for ArrayType in FEM_ArrayTypes
    @eval begin
        function tfqmr!(x::$ArrayType{Tb, 1}, A, b::$ArrayType{Tb, 1}, r::$ArrayType{Tb, 1}; Pl = Identity(), tol::Real, maxiter::Integer, checkiter = 200, kwargs...) where {Tb}
            mul!(r, A, x, -1.)
            r .+= b
            Pl(r)
            (normalized_norm(r) <= tol) && return 0

            iter = 1    
            Lb = length(r)

            alpha = beta = c = one(Tb)
            r0 = FEM_buffer($ArrayType, Tb, Lb)
            r_cgs = FEM_buffer($ArrayType, Tb, Lb)
            p = FEM_buffer($ArrayType, Tb, Lb)
            q = FEM_buffer($ArrayType, Tb, Lb)
            u = FEM_buffer($ArrayType, Tb, Lb)
            v = FEM_buffer($ArrayType, Tb, Lb)
            d = FEM_buffer($ArrayType, Tb, Lb)
            tmp = FEM_buffer($ArrayType, Tb, Lb)

            r0 .= r
            r_cgs .= r
            p .= r
            u .= r
            mul!(v, A, p)
            Pl(v)
            r_norm = r_norm_old = tau = norm(r)
            rho = rhobar = dot(r, r)
            theta = eta = zero(Tb)

            while true
                alpha = rho / dot(v, r0)
                q .= u .- alpha .* v
                v .= u .+ q
                mul!(tmp, A, v)
                Pl(tmp)
                r_cgs .-= alpha .* tmp
                
                r_norm_old = r_norm
                r_norm = norm(r_cgs)

                d .= u .+ (theta ^ 2 * eta / alpha) .* d 
                theta = r_norm_old / tau
                c = 1 / sqrt(1 + theta ^ 2)
                tau *= theta * c
                eta = c ^ 2 * alpha
                x .+= eta .* d

                d .= q .+ (theta ^ 2 * eta / alpha) .* d 
                theta = sqrt(r_norm * r_norm_old) / tau
                c = 1 / sqrt(1 + theta ^ 2)
                tau *= theta * c
                eta = c ^ 2 * alpha
                x .+= eta .* d

                rhobar = rho
                rho = dot(r_cgs, r0)
                beta = rho / rhobar
                u .= r_cgs .+ beta .* q
                p .= u .+ beta .* (q .+ beta .* p)
                mul!(v, A, p)
                Pl(v)

                iter += 1
                (iter > maxiter) && return iter
                if (iter % checkiter) == 0
                    mul!(r, A, x, -1)
                    r .+= b
                    Pl(r)
                    (normalized_norm(r) <= tol) && return iter
                end
            end 
        end
    end
end
