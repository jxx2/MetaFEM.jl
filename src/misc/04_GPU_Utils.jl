macro Dumb_Kernel(body, atom_content)
    major_arr_sym = body.args[end]
    idx_sym = :thread_idx

    total_content = :(begin
    $idx_sym = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if $idx_sym <= size($major_arr_sym)[end]
        $atom_content
    end
    return
    end)

    out_ex = Expr(:function, body, total_content) |> striplines |> stripblocks
    return esc(out_ex)
end

macro Dumb_CUDA_Batch(block_size, body)
    major_arr_sym = body.args[end]
    out_ex = :(@cuda blocks = ceil(Int, length($major_arr_sym)/$block_size) threads = $block_size $body)
    return esc(out_ex)
end

function sort_COO(n::Integer, K_I, K_J)
    K_length = length(K_I)
    P = Int32.(findall(CUDA.ones(Bool, K_length)) .- 1)
    typeof(P)
    function bufferSize()
        out = Ref{Csize_t}(1)
        CUDA.CUSPARSE.cusparseXcoosort_bufferSizeExt(CUDA.CUSPARSE.handle(), n, n, K_length, K_I, K_J, out)
        return out[]
    end

    CUDA.with_workspace(bufferSize) do buffer
        CUDA.CUSPARSE.cusparseXcoosortByRow(CUDA.CUSPARSE.handle(), n, n, K_length, K_I, K_J, P, buffer)
    end
    return P .+ 1
end

function Base.:*(A::CuSparseMatrixCSR{T}, x::CuVector{T}) where T
    b = CUDA.zeros(T, size(A, 1))
    CUDA.CUSPARSE.mv!('N', one(T), A, x, zero(T), b, 'O')
    return b
end

@Dumb_Kernel inc_Num(num, IDs) begin #num[IDs] .+= 1 where IDs can have repeated values
    this_ID = IDs[thread_idx]
    CUDA.atomic_add!(pointer(num) + sizeof(eltype(num)) * (this_ID - 1), Int32(1)) #Note atomic_inc in CUDA.jl is NOT the one you might assume
end

@Dumb_Kernel dec_Num(num, IDs) begin #num[IDs] .-= 1
    this_ID = IDs[thread_idx]
    CUDA.atomic_sub!(pointer(num) + sizeof(eltype(num)) * (this_ID - 1), Int32(1)) #Note atomic_inc in CUDA.jl is NOT the one you might assume
end