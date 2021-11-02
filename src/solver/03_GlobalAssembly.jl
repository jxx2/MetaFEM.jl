function assemble_Global_Variables(; fem_domain::FEM_Domain)
    @Takeout (workpieces, globalfield) FROM fem_domain

    basicfield_size = 0
    for wp in workpieces
        @Takeout (mesh, local_assembly) FROM wp
        @Takeout (global_cpID, is_occupied) FROM mesh.controlpoints WITH PREFIX c_
        @Takeout (global_cpIDs, is_occupied, controlpoint_IDs) FROM mesh.elements WITH PREFIX el_

        variable_size = mesh.variable_size = sum(c_is_occupied)
        cpIDs = findall(c_is_occupied)
        elIDs = findall(el_is_occupied)
        c_global_cpID[cpIDs] .= (findall(c_is_occupied[c_is_occupied]) .+ basicfield_size)
        el_global_cpIDs[:, elIDs] .= c_global_cpID[el_controlpoint_IDs[:, elIDs]]
        basicfield_size += length(local_assembly.basic_vars) * variable_size
    end

    globalfield.basicfield_size = basicfield_size

    max_time_level = globalfield.max_time_level = workpieces .|> get_MaxTimeSteps |> maximum
    globalfield_size = (max_time_level + 1) * basicfield_size
    globalfield.x = CUDA.zeros(FEM_Float, globalfield_size)
    globalfield.dx = CUDA.zeros(FEM_Float, globalfield_size)
    globalfield.x_star = CUDA.zeros(FEM_Float, globalfield_size)
    globalfield.residue = CUDA.zeros(FEM_Float, basicfield_size)

    println("Global field assembled with DOF = ", basicfield_size, " and occupied memory = ", (3 * globalfield_size + basicfield_size) * sizeof(FEM_Float), " bytes")
    assemble_X(workpieces, globalfield)

    assemble_SparseID(workpieces, globalfield)
end

function assemble_X(workpieces::Vector{WorkPiece}, globalfield::GlobalField)
    @Takeout (basicfield_size, x) FROM globalfield
    for wp in workpieces
        @Takeout (mesh.controlpoints, mesh.variable_size, local_assembly.local_innervar_infos) FROM wp
        @Takeout (global_cpID, is_occupied) FROM controlpoints WITH PREFIX c_
        local_data = get_Data(controlpoints)
        cpIDs = findall(c_is_occupied)
        basic_cpID = c_global_cpID[cpIDs]
        Threads.@threads for (local_sym, basic_pos, td_order) in local_innervar_infos
            global_cpIDs = basic_cpID .+ (basic_pos * variable_size + td_order * basicfield_size)
            x[global_cpIDs] .= local_data[local_sym][cpIDs]
        end
    end
end

function dessemble_X(workpieces::Vector{WorkPiece}, globalfield::GlobalField)
    @Takeout (basicfield_size, x) FROM globalfield
    for wp in workpieces
        @Takeout (mesh.controlpoints, mesh.variable_size, local_assembly.local_innervar_infos) FROM wp 
        @Takeout (global_cpID, is_occupied) FROM controlpoints WITH PREFIX c_
        local_data = get_Data(controlpoints)
        cpIDs = findall(c_is_occupied)
        basic_cpID = c_global_cpID[cpIDs]
        Threads.@threads for (local_sym, basic_pos, td_order) in local_innervar_infos
            global_cpIDs = basic_cpID .+ (basic_pos * variable_size + td_order * basicfield_size)
            local_data[local_sym][cpIDs] .= x[global_cpIDs]
        end
    end
end

function assemble_SparseID(workpieces::Vector{WorkPiece}, globalfield::GlobalField)
    last_sparse_ID = 0
    for wp in workpieces
        @Takeout (mesh, local_assembly) FROM wp
        @Takeout (sparse_IDs_by_el, controlpoint_IDs, is_occupied) FROM mesh.elements

        elIDs = findall(is_occupied)
        elnum = length(elIDs)
        el_cp_num = size(controlpoint_IDs, 1)

        local_assembly.sparse_entry_ID = last_sparse_ID
        cp_cp_2_sparseID = construct_GPUDict([:sparseID => Int32(0)])

        cp_cp_keys = CUDA.zeros(UInt64, el_cp_num * el_cp_num * elnum)
        counter = 0
        for i = 1:el_cp_num
            for j = 1:el_cp_num
                dict_ids = (counter * elnum + 1):((counter + 1) * elnum)
                cp_cp_keys[dict_ids] .= I32I32_To_UI64.(controlpoint_IDs[i, elIDs], controlpoint_IDs[j, elIDs])
                counter += 1
            end
        end
        # overlapping_dict_IDs = GPUDict_SetID(cp_cp_2_sparseID, cp_cp_keys)
        GPUDict_SetID(cp_cp_2_sparseID, cp_cp_keys)

        total_dict_IDs = get_Total_IDs(cp_cp_2_sparseID)
        this_unitsize = local_assembly.sparse_unitsize = length(total_dict_IDs)
        local_sparse_ids = findall(CUDA.ones(Bool, this_unitsize)) .+ last_sparse_ID

        sorted_dict_IDs = GPUDict_GetID(cp_cp_2_sparseID, sort(cp_cp_2_sparseID.keys[total_dict_IDs]))
        cp_cp_2_sparseID.sparseID[sorted_dict_IDs] .= local_sparse_ids

        counter = 0
        for i = 1:el_cp_num
            for j = 1:el_cp_num
                cp_cp_keys = I32I32_To_UI64.(controlpoint_IDs[i, elIDs], controlpoint_IDs[j, elIDs])
                local_dict_IDs = GPUDict_GetID(cp_cp_2_sparseID, cp_cp_keys)
                sparse_IDs_by_el[i, j, elIDs] .= cp_cp_2_sparseID.sparseID[local_dict_IDs]
            end
        end
        mesh.cp_cp_2_sparseID = cp_cp_2_sparseID
        last_sparse_ID += length(local_assembly.sparse_mapping) * this_unitsize
    end

    println("Allocating sparse matrix with size = ", last_sparse_ID * (3 * sizeof(FEM_Int) + 2 * sizeof(FEM_Float)), " bytes") #K_I, K_J, K_val_ids, K_linear, K_total
    globalfield.K_I = CUDA.zeros(FEM_Int, last_sparse_ID)
    globalfield.K_J = CUDA.zeros(FEM_Int, last_sparse_ID)
    assemble_KIJ(workpieces, globalfield)

    globalfield.K_val_ids = sort_COO(length(globalfield.residue), globalfield.K_I, globalfield.K_J)
    globalfield.K_linear = CUDA.zeros(FEM_Float, last_sparse_ID)
    globalfield.K_total = CUDA.zeros(FEM_Float, last_sparse_ID)
end

function assemble_KIJ(workpieces::Vector{WorkPiece}, globalfield::GlobalField)
    @Takeout (K_I, K_J) FROM globalfield
    for wp in workpieces
        @Takeout (sparse_entry_ID, sparse_unitsize, sparse_mapping) FROM wp.local_assembly
        @Takeout (controlpoints, variable_size, cp_cp_2_sparseID) FROM wp.mesh

        for ((dual_pos, base_pos), sparse_unit_num) in collect(sparse_mapping)
            sparse_ID_shift = sparse_unit_num * sparse_unitsize
            cpID_shift = (dual_pos, base_pos) .* variable_size #More clear to expand explicitly
            CUDA.@sync @Dumb_CUDA_Batch 256 _assemble_KIJ(cpID_shift, sparse_ID_shift, K_I, K_J, cp_cp_2_sparseID.keys, cp_cp_2_sparseID.sparseID)
        end
    end
end

@Dumb_Kernel _assemble_KIJ(cpID_shift, sparse_ID_shift, K_I, K_J, keys, sparseID) begin
    this_key = keys[thread_idx]
    this_key == 0 && return

    this_sparse_ID = sparseID[thread_idx] + sparse_ID_shift
    dual_shift, base_shift = cpID_shift

    this_cpID = UI64_To_UpperHalf(this_key)
    that_cpID = UI64_To_LowerHalf(this_key)

    K_I[this_sparse_ID] = this_cpID + dual_shift
    K_J[this_sparse_ID] = that_cpID + base_shift
end
