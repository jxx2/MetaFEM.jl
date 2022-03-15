const TABLE_INITIAL_SIZE = 1024

abstract type Abstract_Table{ArrayType} end
get_Data(this_table::Abstract_Table) = getfield(this_table, :data)
Base.getproperty(this_table::Abstract_Table, key::Symbol) = get_Data(this_table)[key]
Base.setproperty!(this_table::Abstract_Table, key::Symbol, value) = setindex!(get_Data(this_table), value, key)

struct FEM_Table{ArrayType} <: Abstract_Table{ArrayType}
    data::Dict{Symbol, ArrayType}
end

construct_FEM_Table(example_typed, example_direct::Vector = []) = construct_FEM_Table(DEFAULT_ARRAYINFO._type, example_typed, example_direct)
function construct_FEM_Table(::Type{ArrayType}, example_typed::ExampleType, example_direct::Vector = []) where {ArrayType, ExampleType}
    names_direct, values_direct = getindex.(example_direct, 1), getindex.(example_direct, 2)
    names_typed = Symbol[name for name in fieldnames(ExampleType) if ~(name in names_direct)]
    values_typed = Any[getfield(example_typed, name) for name in names_typed]

    if ExampleType <: AbstractArray
        push!(names_typed, :arr)
        push!(values_typed, example_typed)
    end

    data_names = Symbol[names_direct..., names_typed..., :is_occupied]
    data_values = Any[values_direct..., values_typed..., false]

    data_vectors = [dvalue isa AbstractArray ? FEM_zeros(ArrayType, eltype(dvalue), size(dvalue)..., TABLE_INITIAL_SIZE) : FEM_zeros(ArrayType, typeof(dvalue), TABLE_INITIAL_SIZE) for dvalue in data_values]
    return FEM_Table{ArrayType}(Dict(data_names .=> data_vectors))
end

function extend!(this_table::FEM_Table{ArrayType}, final_size::Integer) where {ArrayType}
    _data = get_Data(this_table)
    allocated_IDs = allocated(this_table) 

    for (name, arr) in _data
        if ndims(arr) == 1
            new_arr = FEM_zeros(ArrayType, eltype(arr), final_size)
            new_arr[allocated_IDs] .= arr[allocated_IDs]
        else
            effective_dims = size(arr)[1:(end - 1)]
            effective_ranges = Colon().(1, effective_dims)
            new_arr = FEM_zeros(ArrayType, eltype(arr), effective_dims..., final_size)
            new_arr[effective_ranges..., allocated_IDs] .= arr[effective_ranges..., allocated_IDs]
        end            
        _data[name] = new_arr
    end
end

available(this_table::FEM_Table) = findall(.~(this_table.is_occupied))
allocated(this_table::FEM_Table) = findall(this_table.is_occupied)
tablelength(this_table::FEM_Table) = length(this_table.is_occupied)
Base.deleteat!(this_table::FEM_Table, IDs) = (this_table.is_occupied[IDs] .= false; return this_table)

function allocate_by_length!(this_table::FEM_Table, allocating_number::Integer)
    available_length = sum(.~(this_table.is_occupied))
    allocating_number > available_length && extend!(this_table, Int(ceil((allocating_number - available_length) / tablelength(this_table) + 1) * tablelength(this_table)))

    available_IDs = available(this_table)
    allocating_IDs = available_IDs[1:allocating_number]
    this_table.is_occupied[allocating_IDs] .= true
    return allocating_IDs
end
