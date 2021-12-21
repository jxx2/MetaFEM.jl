GPU_DeviceArray{T, N} = CuArray{T, N, CUDA.Mem.DeviceBuffer} where {T, N} 
GPU_UnifiedArray{T, N} = CuArray{T, N, CUDA.Mem.UnifiedBuffer} where {T, N}

mutable struct ArrayDescriptor
    _type::Type
end
const DEFAULT_ARRAYINFO = ArrayDescriptor(GPU_DeviceArray)
# const DEFAULT_ARRAYINFO = ArrayDescriptor(GPU_UnifiedArray)

for src_func in (:zeros, :ones, :rand)
    FEM_func = Symbol("FEM_$(src_func)")
    @eval begin
        $FEM_func(element_type, dims...) = $FEM_func(DEFAULT_ARRAYINFO._type, element_type, dims...)
        $FEM_func(::Type{Array}, ::Type{ElementType}, dims::Number...) where ElementType = $src_func(ElementType, dims...)
        $FEM_func(::Type{GPU_DeviceArray}, ::Type{ElementType}, dims::Number...) where ElementType = (CUDA.$src_func)(ElementType, dims...)
        $FEM_func(::Type{GPU_UnifiedArray}, ::Type{ElementType}, dims::Number...) where ElementType = cu($src_func(ElementType, dims...); unified = true)
    end
end

FEM_ArrayTypes = (:Array, :GPU_DeviceArray, :GPU_UnifiedArray)

FEM_convert(src) = FEM_convert(DEFAULT_ARRAYINFO._type, src)
FEM_convert(::Type{Array}, src::Array) = src
FEM_convert(::Type{Array}, src) = collect(src)
FEM_convert(::Type{GPU_DeviceArray}, src::CuArray) = src
FEM_convert(::Type{GPU_DeviceArray}, src) = cu(src)
FEM_convert(::Type{GPU_UnifiedArray}, src::CuArray) = src
FEM_convert(::Type{GPU_UnifiedArray}, src) = cu(src; unified = true)

function rewrite_Title_ArgTypes(ex::Expr, t_mapping::Dict = Dict())
    for i = 2:length(ex.args)
        if (ex.args[i] isa Expr) && (ex.args[i].head == :(::))
            original_type_sym = ex.args[i].args[2]
            if original_type_sym in keys(t_mapping)
                ex.args[i].args[2] = t_mapping[original_type_sym]
            else
                ex.args[i] = ex.args[i].args[1]
            end
        end
    end
    ex
end

const GPU_Kernel_Generator = PrefixGenerator("GPU")
const GPU_BLOCK_SIZE = 256
function GPU_Single_Kernel(title, atom_content)
    typed_func_title = rewrite_Title_ArgTypes(deepcopy(title), Dict(:Array => :CuArray))
    untyped_func_title = rewrite_Title_ArgTypes(deepcopy(title))

    untyped_func_title.args[1] = GPU_Kernel_Generator(untyped_func_title.args[1])
    major_arr_sym = untyped_func_title.args[end]

    func_ex = Expr(:function, typed_func_title, :(begin
        @cuda blocks = ceil(Int, size($major_arr_sym)[end] / $GPU_BLOCK_SIZE) threads = $GPU_BLOCK_SIZE $untyped_func_title
    end))
        
    kernel_ex = Expr(:function, untyped_func_title, :(begin 
        thread_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if thread_idx <= size($major_arr_sym)[end]
            $atom_content
        end
        return
    end))
    return func_ex, kernel_ex
end

macro Dumb_GPU_Kernel(title, atom_content)
    func_ex, kernel_ex = GPU_Single_Kernel(title, atom_content)
    esc(:(begin
        $kernel_ex
        $func_ex
    end)) 
end

@Dumb_GPU_Kernel inc_Num!(num::Array, vals::Array, IDs::Array) begin #num[IDs] .+= vals where IDs can have repeated values
    CUDA.@atomic num[IDs[thread_idx]] += vals[thread_idx]
    # CUDA.atomic_add!(pointer(num) + sizeof(eltype(num)) * (IDs[thread_idx] - 1), eltype(num)(vals[thread_idx])) 
end

@Dumb_GPU_Kernel inc_Num!(num::Array, this_val, IDs::Array) begin #num[IDs] .+= this_val where IDs can have repeated values
    CUDA.@atomic num[IDs[thread_idx]] += this_val
    # CUDA.atomic_add!(pointer(num) + sizeof(eltype(num)) * (IDs[thread_idx] - 1), eltype(num)(this_val)) 
end

function sort_COO!(n::Integer, K_I, K_J) # need to rewrite by distributed bitonic ones
    K_length = length(K_I)
    P = Int32.(findall(CUDA.ones(Bool, K_length)) .- 1) #!!!may need to change
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

for ArrayType in FEM_ArrayTypes
    @eval begin
        function generate_J_ptr(Is::$ArrayType{FEM_Int}, m::Integer)
            J_ptr = FEM_zeros($ArrayType, FEM_Int, m + 1)
            J_ptr .= length(Is) + 1
            
            compress_CSR!(J_ptr, Is)
            return J_ptr
        end
    end
end

shifted_Pointer(arr, offset) = pointer(arr) + sizeof(eltype(arr)) * (offset - 1)
@Dumb_GPU_Kernel compress_CSR!(J_ptr::Array, Is::Array) begin 
    CUDA.atomic_min!(shifted_Pointer(J_ptr, Is[thread_idx]), eltype(J_ptr)(thread_idx)) 
end

mul!(b::CuVector{T}, A::CuSparseMatrixCSR{T}, x::CuVector{T}, alpha::Number = 1., beta::Number = 0.) where T = CUDA.CUSPARSE.mv!('N', T(alpha), A, x, T(beta), b, 'O')
tmul!(b::CuVector{T}, A::CuSparseMatrixCSR{T}, x::CuVector{T}, alpha::Number = 1., beta::Number = 0.)  where T = CUDA.CUSPARSE.mv!('T', T(alpha), A, x, T(beta), b, 'O')

function Base.:*(A::CuSparseMatrixCSR{T}, x::CuVector{T}) where T #!!!may need to change
    b = CUDA.zeros(T, size(A, 1))
    mul!(b, A, x)
    return b
end

for A in (CUDA.Mem.DeviceBuffer, CUDA.Mem.UnifiedBuffer)
    for B in (CUDA.Mem.DeviceBuffer, CUDA.Mem.UnifiedBuffer)
        (A == CUDA.Mem.DeviceBuffer) && (B == CUDA.Mem.DeviceBuffer) && continue
        @eval begin
            function Base.unsafe_copyto!(dest::CuArray{T,<:Any, $A}, doffs,
                                        src::CuArray{T,<:Any, $B}, soffs, n) where T
            CUDA.context(dest) == CUDA.context(src) || throw(ArgumentError("copying between arrays from different contexts"))
            CUDA.@context! CUDA.context(dest) begin
                GC.@preserve src dest begin
                unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
                if Base.isbitsunion(T)
                    unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
                end
                end
            end
            return dest
            end
        end
    end
end