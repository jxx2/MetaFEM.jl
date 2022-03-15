"""
    iterative_Solve!(globalfield::GlobalField; Sv_func!::Function = bicgstabl_GS!, Pr_func!::Function = Pr_Jacobi!, Pl_func::Function = Identity, max_pass = 4, kwargs...)

Wrapper of general iterative solvers, where:

* `Sv_func!` is the detailed iterative solver to be called with the folloiwng choices:
    * BICG family: `bicgstabl_GS!`, `bicgstabl! ``
    * IDRS family: `idrs!`, `idrs_original!` 
    * LSQR: `lsqr!`
    * GMRES: `gmres!` 
* `Pr_func` is the right preconditioner function with the following choices:
    * `Identity`: do nothing.
    * `Pr_Jacobi` (default): normalize each column by the diagnal element or the column norm. 
* `Pl_func` is the left preconditioner function with the following choices:
    * `Identity` (default): do nothing.
    * `Pl_Jacobi`: normalize each row by the diagnal element or the row norm. 
    * `Pl_ILU`: ILU preconditioner provided by ilu02! in CUDA.jl.
* `max_pass` is the number of attempts in one solving step, e.g., let max_pass = 4 and let maxiter = 3000 in idrs!, then the solver will run 4 (restarted) batches of 3000 iterations in solving every `Kx=d`.
* Other `kwargs...` will be passed into `Sv_func!` as keyword arguments.

Note, by default the iterative solver will be right Jacobi preconditioned only. 
A right preconditioner modifies the matrix once when the solver begins and modifies the x once when the solver exits, i.e., 
roughly cost one more mat-vec mul!(mv!) and one more vec-vec axpy!. The right preconditioner will both ameliorate the numerical inaccuracy and reduce the iteration number from the ill-conditioning.

A left preconditioner modifies the vector every time a mat-vec mul! occurs, and it can only reduce the iteration number but not ameliorate the (condition number induced) numerical inaccuracy.
However, left preconditioner can be more complicated, e.g., ILU, than the right preconditioner, for which only Jacobi can be practically applied when the matrix is large (in author's limited knowledge).
"""
iterative_Solve!

for ArrayType in FEM_ArrayTypes
    @eval begin
        function iterative_Solve!(globalfield::GlobalField{$ArrayType}; Sv_func!::Function = idrs!, Pr_func!::Function = Pr_Jacobi!, Pl_func::Function = Identity, max_pass = 4, kwargs...)
                @Takeout (basicfield_size, K_I, K_J, K_J_ptr, K_val_ids, K_total, residue, converge_tol) FROM globalfield

            K_vals = K_total[K_val_ids]
            A = FEM_SpMat_CSR(K_J_ptr, K_J, K_vals, (basicfield_size, basicfield_size))
            
            Pr = Pr_func!(A)

            Pl = Pl_func(A)

            b = residue
            r = copy(b)

            x = FEM_zeros($ArrayType, FEM_Float, length(residue))
            println("Solver $Sv_func! with initial res = $(normalized_norm(b)) and target res = $converge_tol")
            println(typeof(x))
            pass_number = 1
            tol_factor = 1.
            while true
                pass_iters = Sv_func!(x, A, b, r; Pl = Pl, tol = tol_factor * converge_tol, kwargs...)

                mul!(r, A, x, -1.)
                r .+= b
                # r .= b .- A * x # re-compute r, since in the local iterative solver r may be indirectly calculated in an incremental way and inaccurate
                res = normalized_norm(r)

                if Pl_func != Identity # If left preconditioned, residue need to be corrected
                    preconditioned_res = normalized_norm(Pl(r)) 
                    tol_factor = min(preconditioned_res / res, 1.) 
                    println("pass $pass_number with res = $res preconditioned res = $preconditioned_res iter = $pass_iters.")
                else
                    println("pass $pass_number with res = $res iter = $pass_iters.")
                end

                if (res < converge_tol) || (pass_number >= max_pass)
                    println("break")
                    break
                else
                    println("next pass preconditioned res threshold = $(tol_factor * converge_tol).")
                    pass_number += 1
                end
            end

            return Pr(x)
        end
    end
end

struct _Identity end
"""
    Identity(args...; kwargs...)

Returns an identity preconditioner, i.e., do nothing at all.
"""
Identity(args...; kwargs...) = _Identity()
(::_Identity)(b) = b

struct _JacobiP
    jac_vec
end

function (P::_JacobiP)(b) 
    b ./= P.jac_vec
    return b
end

"""
    Pr_Jacobi!(A::CuSparseMatrixCSR{T}; normalized_by_column = false)

GPU Jacobi right preconditioner. Each column of the matrix will be normalized, if `normalized_by_column` = `false` by the diagonal element, otherwise, by the column norm. 
"""
function Pr_Jacobi!(A::CuSparseMatrixCSR{T}; normalized_by_column = false) where T
    J_ptr = A.rowPtr
    Js = A.colVal
    Ks = A.nzVal

    # jac_vec = CUDA.ones(T, size(A, 2))
    if normalized_by_column 
        jac_vec = CUDA.zeros(T, size(A, 2))
        Jacobi2_By_Colomn(J_ptr, Js, Ks, jac_vec) 
        jac_vec .^= 0.5
    else 
        jac_vec = CUDA.ones(T, size(A, 2))
        Jacobi_By_Diagonal(J_ptr, Js, Ks, jac_vec)
    end

    Mat_Div_Jacobi(J_ptr, Js, Ks, jac_vec) # right preconditioner will modify the matrix
    return _JacobiP(jac_vec)
end

@Dumb_GPU_Kernel Jacobi_By_Diagonal(J_ptr::Array, Js::Array, Ks::Array, jac_vec::Array) begin
    j_start_pos = J_ptr[thread_idx]
    j_final_pos = J_ptr[thread_idx + 1] - 1
    for j_pos = j_start_pos:j_final_pos 
        if Js[j_pos] == thread_idx
            jac_vec[thread_idx] = abs(Ks[j_pos])
        end
    end
end

@Dumb_GPU_Kernel Jacobi2_By_Colomn(J_ptr::Array, Js::Array, Ks::Array, jac_vec::Array) begin #note in this kernel the final sqrt is left outside because my syncronization does not work, sad
    j_start_pos = J_ptr[thread_idx]
    j_final_pos = J_ptr[thread_idx + 1] - 1
    for j_pos = j_start_pos:j_final_pos
        this_J = Js[j_pos]
        CUDA.@atomic jac_vec[this_J] += Ks[j_pos] ^ 2
    end
end

@Dumb_GPU_Kernel Mat_Div_Jacobi(J_ptr::Array, Js::Array, Ks::Array, jac_vec::Array) begin
    j_start_pos = J_ptr[thread_idx]
    j_final_pos = J_ptr[thread_idx + 1] - 1
    for j_pos = j_start_pos:j_final_pos 
        this_J = Js[j_pos]
        Ks[j_pos] /= jac_vec[this_J]
    end
end

"""
    Pl_Jacobi(A::CuSparseMatrixCSR{T}; normalized_by_row = false)

GPU Jacobi left preconditioner. Each row of the matrix will be normalized through modifying the final residue vector instead of the matrix, if `normalized_by_row` = `false`, by the diagonal element, otherwise, by the row norm. 
"""
function Pl_Jacobi(A::CuSparseMatrixCSR{T}; normalized_by_row = false) where T
    J_ptr = A.rowPtr
    Js = A.colVal
    Ks = A.nzVal

    if normalized_by_row 
        jac_vec = CUDA.zeros(T, size(A, 2))
        Jacobi_By_Row(J_ptr, Js, Ks, jac_vec)
    else 
        jac_vec = CUDA.ones(T, size(A, 2))
        Jacobi_By_Diagonal(J_ptr, Js, Ks, jac_vec)
    end
    return _JacobiP(jac_vec)
end

@Dumb_GPU_Kernel Jacobi_By_Row(J_ptr::Array, Js::Array, Ks::Array, jac_vec::Array) begin
    j_start_pos = J_ptr[thread_idx]
    j_final_pos = J_ptr[thread_idx + 1] - 1
    for j_pos = j_start_pos:j_final_pos
        jac_vec[thread_idx] += Ks[j_pos] ^ 2
    end
    jac_vec[thread_idx] = sqrt(jac_vec[thread_idx])
end

struct _Pl_ILU
    ilu_mat
end

"""
    Pl_ILU(A)

GPU ILU preconditioner with `CUDA`'s `ilu02!`.
"""
Pl_ILU(A) = _Pl_ILU(ilu02!(copy(A), 'O'))

function (P::_Pl_ILU)(b)
    LinearAlgebra.ldiv!(UnitLowerTriangular(P.ilu_mat), b)
    LinearAlgebra.ldiv!(UpperTriangular(P.ilu_mat), b)
    return b
end

# TODO _Pl_SPAI