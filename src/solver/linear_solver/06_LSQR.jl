"""
    lsqr!(x, A, b, r; Pl = Identity(), tol::Real, maxiter::Integer, s = 20, kwargs...)

LSQR solver, robust over most matrices, e.g., the ones generated randomly by `sprand`, where solvers like IDRs may explode. However, IDRs/... may converge much faster than LSQR when stable.
"""
lsqr!

for ArrayType in FEM_ArrayTypes
    @eval begin
        function lsqr!(x::$ArrayType{Tb, 1}, A, b::$ArrayType{Tb, 1}, r::$ArrayType{Tb, 1}; Pl = Identity(), tol::Real, maxiter::Integer, kwargs...) where {Tb}
            mul!(r, A, x, -1.)
            r .+= b
            Pl(r)
            (normalized_norm(r) <= tol) && return 0
            
            iter = 1

            u = copy(r) #note u begin with r, not b
            beta = norm(u)
            u ./= beta #with previous check u cant be 0

            v = copy(u)
            tmul!(v, A, u) 
            Pl(v)
            alpha = norm(v)
            if alpha != 0
                v ./= alpha
            end

            w = copy(v)
            phibar = beta
            rhobar = alpha

            tmp = copy(u) # buffer for left precondition 
            while true
                mul!(tmp, A, v)
                u .= Pl(tmp) .- (alpha .* u)

                beta = norm(u)
                if beta != 0
                    u ./= beta
                    tmul!(tmp, A, u)
                    v .= Pl(tmp) .- (beta .* v)

                    alpha = norm(v)
                    if alpha != 0
                        v ./= alpha
                    end
                end

                rho = sqrt(abs2(rhobar) + abs2(beta))
                c = rhobar / rho
                s = beta / rho
                theta = s * alpha
                rhobar = - c * alpha
                phi = c * phibar
                phibar = s * phibar

                x .+= (phi / rho) .* w
                w .= v .- (theta / rho) .* w

                iter += 1

                mul!(r, A, x, -1)
                r .+= b
                Pl(r)
                ((normalized_norm(r) <= tol) || (iter > maxiter)) && return iter
            end 
        end
    end
end


