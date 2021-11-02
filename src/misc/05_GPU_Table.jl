const BATCH_SIZE = 8

abstract type Abstract_Table end
get_Data(s::Abstract_Table) = getfield(s, :data)
Base.getproperty(s::Abstract_Table, key::Symbol) = get_Data(s)[key]
Base.setproperty!(s::Abstract_Table, key::Symbol, value) = setindex!(get_Data(s), value, key)
volumeof(s::Abstract_Table) = sum([sizeof(item) for item in values(get_Data(s))])

struct GPUTable <: Abstract_Table
    data::Dict{Symbol, CuArray}
end

function construct_GPUTable(example_typed::T, example_direct::Vector = []) where {T}
    names_direct, values_direct = getindex.(example_direct, 1), getindex.(example_direct, 2)
    names_typed = Symbol[name for name in fieldnames(T) if ~(name in names_direct)]
    values_typed = [getfield(example_typed, name) for name in names_typed]

    data_names = Symbol[names_direct..., names_typed..., :is_occupied]
    data_values = Any[values_direct..., values_typed..., false]

    if T <: AbstractArray
        push!(data_names, :arr)
        push!(data_values, example_typed)
    end
    data_vectors = [dvalue isa AbstractArray ? CUDA.zeros(eltype(dvalue), tuple(size(dvalue)..., BATCH_SIZE)) : CUDA.zeros(typeof(dvalue), BATCH_SIZE)
                for dvalue in data_values]
    return (data_names .=> data_vectors) |> Dict |> GPUTable
end

available(s::GPUTable) = findall(.~(s.is_occupied))
allocated(s::GPUTable) = findall(s.is_occupied)
tablelength(s::GPUTable) = length(s.is_occupied)
Base.deleteat!(s::GPUTable, IDs) = (s.is_occupied[IDs] .= false; return s)
function extend!(s::GPUTable, batchsize::Integer)
    _data = get_Data(s)
    for (name, arr) in _data
        arr_size, arr_dim, arr_type = size(arr), ndims(arr), eltype(arr)
        new_arr_size = arr_dim == 1 ? batchsize : tuple(arr_size[1:(end - 1)]..., batchsize)
        new_arr = CUDA.zeros(arr_type, new_arr_size...)
        _data[name] = cat(arr, new_arr, dims = arr_dim)
    end
end

function allocate_by_length!(s::GPUTable, allocating_number::Integer; batchsize::Integer = BATCH_SIZE)
    available_length = sum(.~(s.is_occupied))
    allocating_number > available_length && extend!(s, FEM_Int(ceil((allocating_number - available_length) / batchsize) * batchsize))

    CUDA.@sync available_IDs = available(s)
    allocating_IDs = available_IDs[1:allocating_number]
    s.is_occupied[allocating_IDs] .= true
    return allocating_IDs
end
