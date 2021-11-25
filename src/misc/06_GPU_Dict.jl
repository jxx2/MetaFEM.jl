#Thomas Wang's hashing, the same as julia base
function GPU_hash_64_64(x::UInt64)
    a = x
    a = ~a + a << 21
    a =  a ⊻ a >> 24
    a =  a + a << 3 + a << 8
    a =  a ⊻ a >> 14
    a =  a + a << 2 + a << 4
    a =  a ⊻ a >> 28
    a =  a + a << 31
    return a
end
_DictSize(x::Integer) = x < (16 * 2 / 3) ? 16 : one(x)<<((sizeof(x)<<3)-leading_zeros(Int(ceil((x * 1.5))) - 1))

struct GPUDict <: Abstract_Table
    data::Dict{Symbol, CuArray}
    var_names::Vector{Symbol}
end

get_VarNames(s::GPUDict) = getfield(s, :var_names)
get_Total_IDs(s::GPUDict) = findall(s.keys .!= 0) #Note dict key cannot be 0
tablelength(s::GPUDict) = length(s.keys)
function construct_GPUDict(data_external::Vector = [], size_hint::Integer = 16)
    data_internal = [:hash_init, :hash_prev, :hash_next] .=> Int32(0)
    names_internal, values_internal = getindex.(data_internal, 1), getindex.(data_internal, 2)
    names_external, values_external = getindex.(data_external, 1), getindex.(data_external, 2)

    data_names = Symbol[names_external..., :keys, :hashs, names_internal...]
    data_values = Any[values_external..., UInt64(0), UInt64(0), values_internal...]
    dict_size = _DictSize(size_hint)

    data_vectors = [dvalue isa AbstractArray ? CUDA.zeros(eltype(dvalue), tuple(size(dvalue)..., dict_size)) : CUDA.zeros(typeof(dvalue), dict_size) for dvalue in data_values]
    return GPUDict(Dict(data_names .=> data_vectors), names_external)
end

dumb_GPUDict_Init(val) = construct_GPUDict([:vals => zero(typeof(val))], 16)
dumb_GPUDict_Init(new_keys::CuVector{T}, new_vals::CuVector) where T <: Integer = dumb_GPUDict_Init(to_UInt64.(new_keys), new_vals)
function dumb_GPUDict_Init(new_keys::CuVector{UInt64}, new_vals::CuVector) 
    dict_size = new_keys |> length |> _DictSize
    new_dict = construct_GPUDict([:vals => zero(eltype(new_vals))], dict_size)
    new_IDs =  GPUDict_SetID(new_dict, new_keys)
    new_dict.vals[new_IDs] .= new_vals
    return new_dict
end

function GPUDict_SetID(source_dict::GPUDict, new_keys::CuVector{UInt64})
    isempty(new_keys) && return cu(Int32[])

    @Takeout (keys, hashs, hash_init, hash_prev, hash_next) FROM source_dict
    raw_size = length(new_keys)
    dict_size = length(keys)
    source_IDs = get_Total_IDs(source_dict)
    source_dict_size = length(source_IDs)
    estimate_dict_size = _DictSize(raw_size + source_dict_size)
    new_IDs = CUDA.zeros(Int32, raw_size)

    if estimate_dict_size == dict_size
        @Dumb_CUDA_Batch 256 dict_SetID(keys, hashs, hash_init, hash_prev, hash_next, new_keys, new_IDs)
    else
        expanded_keys = CUDA.zeros(UInt64, estimate_dict_size)
        expanded_hashs = CUDA.zeros(UInt64, estimate_dict_size)
        expanded_hash_init = CUDA.zeros(Int32, estimate_dict_size)
        expanded_hash_prev = CUDA.zeros(Int32, estimate_dict_size)
        expanded_hash_next = CUDA.zeros(Int32, estimate_dict_size)

        source_keys = keys[source_IDs]
        mapped_IDs =  CUDA.zeros(Int32, source_dict_size)
        if source_dict_size > 0
            @Dumb_CUDA_Batch 256 dict_SetID(expanded_keys, expanded_hashs, expanded_hash_init, expanded_hash_prev, expanded_hash_next, source_keys, mapped_IDs)
        end
        
        source_data = get_Data(source_dict)
        for var_name in get_VarNames(source_dict)
            var_data = source_data[var_name]
            mapped_data = CUDA.zeros(eltype(var_data), estimate_dict_size)
            mapped_data[mapped_IDs] .= var_data[source_IDs]
            source_data[var_name] = mapped_data
        end
        source_dict.keys = expanded_keys
        source_dict.hashs = expanded_hashs
        source_dict.hash_init = expanded_hash_init
        source_dict.hash_prev = expanded_hash_prev
        source_dict.hash_next = expanded_hash_next

        @Dumb_CUDA_Batch 256 dict_SetID(expanded_keys, expanded_hashs, expanded_hash_init, expanded_hash_prev, expanded_hash_next, new_keys, new_IDs)
    end
    return new_IDs
end

update_TruncID(prev_ID, current_size) = ((prev_ID % Int32) & Int32(current_size - 1)) + Int32(1)
@Dumb_Kernel dict_SetID(dict_keys, hashs, hash_init, hash_prev, hash_next, new_keys, new_IDs) begin
    this_key = new_keys[thread_idx]
    this_hash = GPU_hash_64_64(this_key)

    current_size = length(dict_keys)
    start_ID = update_TruncID(this_hash, current_size)
    if hash_init[start_ID] == 0
        this_ID = start_ID
    else
        this_ID = hash_init[start_ID]
    end
    last_front_ID = Int32(0)
    while true
        local_key = CUDA.atomic_cas!(pointer(dict_keys) + sizeof(eltype(dict_keys)) * (this_ID - 1), UInt64(0), this_key) #Note dict key cannot be 0
        if local_key == 0 #write to new slot
            hashs[this_ID] = this_hash
            if (last_front_ID != 0) # && (last_front_ID != this_ID) #Not necessary
                hash_prev[this_ID] = last_front_ID
                hash_next[last_front_ID] = this_ID
            else
                hash_init[start_ID] = this_ID
            end
            break
        elseif local_key == this_key #overwrite
            break
        elseif update_TruncID(GPU_hash_64_64(local_key), current_size) == start_ID #slightly smarter probing, note hash is swapped by atomic_cas, so work under parallel setindex
            if hash_next[this_ID] == Int32(0)
                last_front_ID = this_ID
                this_ID = update_TruncID(this_ID, current_size)
            else
                this_ID = hash_next[this_ID]
            end
        else
            this_ID = update_TruncID(this_ID, current_size)
        end
    end
    new_IDs[thread_idx] = this_ID
end

function GPUDict_GetID(source_dict::GPUDict, target_keys::CuVector{UInt64})
    @Takeout (keys, hashs, hash_init, hash_next) FROM source_dict
    target_IDs = CUDA.zeros(Int32, length(target_keys))
    @Dumb_CUDA_Batch 256 dict_GetID(keys, hashs, hash_init, hash_next, target_keys, target_IDs)
    return target_IDs
end

dumb_GPUDict_Get(source_dict::GPUDict, target_keys::CuVector{T}) where T <: Integer = dumb_GPUDict_Get(source_dict, to_UInt64.(target_keys))
function dumb_GPUDict_Get(source_dict::GPUDict, target_keys::CuVector{UInt64})
    target_IDs = GPUDict_GetID(source_dict, target_keys)
    target_arr = CUDA.zeros(eltype(source_dict.vals), length(target_IDs))
    
    prod(target_IDs .> 0) || error("Wrong key")
    return source_dict.vals[target_IDs]
end

function GPUDict_GetID_Single(dict_keys, hashs, hash_init, hash_next, target_key)
    current_size = length(dict_keys)
    start_ID = update_TruncID(GPU_hash_64_64(target_key), current_size)
    prob_ID = hash_init[start_ID]
    prob_ID == 0 && return Int32(0)
    this_ID = prob_ID
    while true
        dict_keys[this_ID] == target_key && return this_ID
        update_TruncID(hashs[this_ID], current_size) == start_ID || return Int32(-1)
        hash_next[this_ID] == 0 && return Int32(0)
        this_ID = hash_next[this_ID]
    end
end

# @Dumb_Kernel dict_GetID(dict_keys, hashs, hash_init, hash_next, target_keys, target_IDs) begin
#     target_IDs[thread_idx] = GPUDict_GetID_Single(dict_keys, hashs, hash_init, hash_next, target_keys[thread_idx])
# end

@Dumb_Kernel dict_GetID(dict_keys, hashs, hash_init, hash_next, target_keys, target_IDs) begin
    # target_IDs[thread_idx] = GPUDict_GetID_Single(dict_keys, hashs, hash_init, hash_next, target_keys[thread_idx])
    target_key = target_keys[thread_idx]
    current_size = length(dict_keys)
    start_ID = update_TruncID(GPU_hash_64_64(target_key), current_size)
    this_ID = hash_init[start_ID]

    if this_ID == 0 
        target_IDs[thread_idx] = Int32(0)
        return 
    end

    while true
        if dict_keys[this_ID] == target_key # Found
            target_IDs[thread_idx] = this_ID
            return
        elseif ~(update_TruncID(hashs[this_ID], current_size) == start_ID) #error
            target_IDs[thread_idx] = Int32(-1)
            return
        elseif hash_next[this_ID] == 0 # Not exist
            target_IDs[thread_idx] = Int32(0)
            return
        end
        this_ID = hash_next[this_ID]
    end
end

function GPUDict_DelID(source_dict::GPUDict, del_keys::CuVector{UInt64})
    @Takeout (keys, hashs, hash_init, hash_prev, hash_next) FROM source_dict
    del_infos = GPUDict_GetID(source_dict, del_keys)
    del_IDs = del_infos[del_infos .> 0]
    is_deleted = CUDA.zeros(Bool, length(keys))
    is_deleted[del_IDs] .= true
    #del_vals = vals[del_IDs]
    @Dumb_CUDA_Batch 256 dict_DelID_Reconnect(keys, hashs, hash_init, hash_prev, hash_next, is_deleted)
    keys[del_IDs] .= UInt64(0)
    hashs[del_IDs] .= UInt64(0)
    hash_prev[del_IDs] .= Int32(0)
    hash_next[del_IDs] .= Int32(0)
    return del_IDs
end

function dumb_GPUDict_Del(source_dict::GPUDict, del_keys::CuVector{UInt64})
    delIDs = GPUDict_DelID(source_dict, del_keys)
    source_dict.vals[delIDs] .= 0
end

@Dumb_Kernel dict_DelID_Reconnect(dict_keys, hashs, hash_init, hash_prev, hash_next, is_deleted) begin
    #dict_del_ID = del_IDs[thread_idx]
    if is_deleted[thread_idx]
        this_hash = hashs[thread_idx]
        start_ID = update_TruncID(this_hash, length(dict_keys))

        source_prev_ID = hash_prev[thread_idx]
        source_next_ID = hash_next[thread_idx]
        exist_prev_ID = Int32(0)
        local_prev_ID = source_prev_ID
        while true
            if (local_prev_ID == Int32(0)) || (~is_deleted[local_prev_ID])
                exist_prev_ID = local_prev_ID
                break
            end
            local_prev_ID = hash_prev[local_prev_ID]
        end
        exist_next_ID = Int32(0)
        local_next_ID = source_next_ID
        while true
            if (local_next_ID == Int32(0)) || (~is_deleted[local_next_ID])
                exist_next_ID = local_next_ID
                break
            end
            local_next_ID = hash_next[local_next_ID]
        end

        if source_prev_ID == Int32(0)
            hash_init[start_ID] = exist_next_ID #exist_next_ID = 0 -> not exist -> hash_init = 0, consistent
        end
        if (exist_prev_ID != Int32(0)) && (source_prev_ID == exist_prev_ID) #the first node after exist_prev_ID
            hash_next[exist_prev_ID] = exist_next_ID
        end
        if (exist_next_ID != Int32(0)) && (source_next_ID == exist_next_ID) #the first node before exist_next_ID
            hash_prev[exist_next_ID] = exist_prev_ID
        end
    end
end
to_UInt64(x) = Int(x) % UInt64
to_UInt64(x::Integer) = x % UInt64
to_UInt64_32(x) = to_UInt64(x) & (UInt32(0) - UInt32(1))
to_UInt64_30(x) = to_UInt64(x) & ((UInt32(0) - UInt32(1)) >> 2)
to_UInt64_20(x) = to_UInt64(x) & ((UInt32(0) - UInt32(1)) >> 12)

I32I32_To_UI64(x, y) = (to_UInt64_32(x) << 32) + to_UInt64_32(y)
UI64_To_UpperHalf(x::UInt64) = ((x >> 32) & (UInt32(0) - UInt32(1))) % Int32
UI64_To_LowerHalf(x::UInt64) = (x & (UInt32(0) - UInt32(1))) % Int32

I4I30I30_To_UI64(x, y, z) = (to_UInt64(x + 1) << 60) + (to_UInt64_30(y) << 30) + to_UInt64_30(z) # when 0, 0, 0 key should not be 0
I4I20I20I20_To_UI64(x, y, z, w) = (to_UInt64(x + 1) << 60) + (to_UInt64_20(y) << 40) + (to_UInt64_20(z) << 20) + to_UInt64_20(w) # when 0, 0, 0 key should not be 0

UI64_To_64_60(x::UInt64) = ((x >> 60) & (UInt32(0) - UInt32(1))) % Int32 - 1
UI64_To_60_30(x::UInt64) = Int32(((x << 4) % Int64) >> 34) 
UI64_To_30_00(x::UInt64) = Int32(((x << 34) % Int64) >> 34)
UI64_To_60_40(x::UInt64) = Int32(((x << 4) % Int64) >> 44) 
UI64_To_40_20(x::UInt64) = Int32(((x << 24) % Int64) >> 44)
UI64_To_20_00(x::UInt64) = Int32(((x << 44) % Int64) >> 44)