@Dumb_GPU_Kernel _Var_Basic(itp_vals::Array, sd_IDs, cpID_shift, el_g_cpIDs::Array, x_star::Array, target::Array, itg_hostIDs::Array, elIDs::Array) begin #Note this is a sum on itp, not itg
    itg_func_num = size(itp_vals, 1)
    itp_num = size(itp_vals, 2)

    this_elID = elIDs[thread_idx]
    this_itghostID = itg_hostIDs[thread_idx]
    for this_base_id = 1:itp_num
        this_cp_ID = el_g_cpIDs[this_base_id, this_elID] + cpID_shift
        for this_itg_id = 1:itg_func_num
            CUDA.@atomic target[this_itg_id, thread_idx] += itp_vals[this_itg_id, this_base_id, sd_IDs..., this_itghostID] * x_star[this_cp_ID]
        end
    end
end

@Dumb_GPU_Kernel _Var_Cut(itp_vals::Array, sd_IDs, cpID_shift, el_g_cpIDs::Array, x_star::Array, target::Array, elIDs::Array) begin #Note this is a sum on itp, not itg
    itg_func_num = size(itp_vals, 1)
    itp_num = size(itp_vals, 2)

    this_elID = elIDs[thread_idx]
    for this_base_id = 1:itp_num
        this_cp_ID = el_g_cpIDs[this_base_id, this_elID] + cpID_shift
        for this_itg_id = 1:itg_func_num
            CUDA.@atomic target[this_itg_id, thread_idx] += itp_vals[this_itg_id, this_base_id, sd_IDs...] * x_star[this_cp_ID]
        end
    end
end

@Dumb_GPU_Kernel _Kval_Basic(itp_vals::Array, dual_sd_IDs, base_sd_IDs, vals::Array, sparse_IDs_by_el::Array, sparse_ID_shift, K_val::Array, itg_hostIDs::Array, elIDs::Array) begin #Note all local
    itg_func_num = size(itp_vals, 1)
    itp_num = size(itp_vals, 2)

    this_elID = elIDs[thread_idx]
    this_itghostID = itg_hostIDs[thread_idx]
    for this_dual_id = 1:itp_num
        for this_base_id = 1:itp_num
            this_sparse_ID = sparse_IDs_by_el[this_dual_id, this_base_id, this_elID] + sparse_ID_shift
            sum = 0.
            for this_itg_id = 1:itg_func_num
                sum += itp_vals[this_itg_id, this_dual_id, dual_sd_IDs..., this_itghostID] * 
                       itp_vals[this_itg_id, this_base_id, base_sd_IDs..., this_itghostID] * vals[this_itg_id, thread_idx]
            end
            CUDA.@atomic K_val[this_sparse_ID] += sum
        end
    end
end

@Dumb_GPU_Kernel _Kval_Cut(itp_vals::Array, dual_sd_IDs, base_sd_IDs, vals::Array, sparse_IDs_by_el::Array, sparse_ID_shift, K_val::Array, elIDs::Array) begin #Note all local
    itg_func_num = size(itp_vals, 1)
    itp_num = size(itp_vals, 2)

    this_elID = elIDs[thread_idx]
    for this_dual_id = 1:itp_num
        for this_base_id = 1:itp_num
            this_sparse_ID = sparse_IDs_by_el[this_dual_id, this_base_id, this_elID] + sparse_ID_shift
            sum = 0.
            for this_itg_id = 1:itg_func_num
                sum += itp_vals[this_itg_id, this_dual_id, dual_sd_IDs...] * 
                       itp_vals[this_itg_id, this_base_id, base_sd_IDs...] * vals[this_itg_id, thread_idx]
            end
            CUDA.@atomic K_val[this_sparse_ID] += sum
        end
    end
end

@Dumb_GPU_Kernel _Res_Basic(itp_vals::Array, dual_sd_IDs, vals::Array, cpID_shift, el_g_cpIDs::Array, residue::Array, itg_hostIDs::Array, elIDs::Array) begin
    itg_func_num = size(itp_vals, 1)
    itp_num = size(itp_vals, 2)

    this_elID = elIDs[thread_idx]
    this_itghostID = itg_hostIDs[thread_idx]
    for this_dual_id = 1:itp_num
        this_cp_ID = el_g_cpIDs[this_dual_id, this_elID]
        sum = 0.
        for this_itg_id = 1:itg_func_num
            sum += itp_vals[this_itg_id, this_dual_id, dual_sd_IDs..., this_itghostID] * vals[this_itg_id, thread_idx]
        end
        CUDA.@atomic residue[this_cp_ID + cpID_shift] += sum
    end
end

@Dumb_GPU_Kernel _Res_Cut(itp_vals::Array, dual_sd_IDs, vals::Array, cpID_shift, el_g_cpIDs::Array, residue::Array, elIDs::Array) begin
    itg_func_num = size(itp_vals, 1)
    itp_num = size(itp_vals, 2)

    this_elID = elIDs[thread_idx]
    for this_dual_id = 1:itp_num
        this_cp_ID = el_g_cpIDs[this_dual_id, this_elID]
        sum = 0.
        for this_itg_id = 1:itg_func_num
            sum += itp_vals[this_itg_id, this_dual_id, dual_sd_IDs...] * vals[this_itg_id, thread_idx]
        end
        CUDA.@atomic residue[this_cp_ID + cpID_shift] += sum
    end
end
