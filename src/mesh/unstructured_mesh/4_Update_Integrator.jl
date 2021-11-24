gen_Kernel_Name(prefix::String, max_sd_order::Integer, dim::Integer) = Symbol(string(prefix, max_sd_order, "_", dim, "D"))
function gen_BasicDomain_Funcs(dim::Integer)    
    ID_for_no_diff = fill(1, dim)
    jac_block = :(begin
        X_IDs = ntuple(x -> x == X_dim ? 2 : 1, $dim)
    end)
    for this_dim = 1:dim
        x_sym = Symbol(string("x", this_dim))
        push!(jac_block.args, :(jacobian[$this_dim, X_dim, :, elIDs] .= ref_itp_vals[:, :, X_IDs...] * controlpoints.$x_sym[cpIDs]))
    end

    invJac_kernel = Symbol(string("inv_Jac_", dim, "D"))
    kernel_libs = Symbol(string("BASE_KERNELS_", dim, "D"))
    func = Symbol(string("update_BasicElements_", dim, "D"))
    ex = :(function ($func)(mesh::FEM_WP_Mesh, this_space::Classical_Discretization, itpval_kernel::Function)
        @Takeout (controlpoints, elements) FROM mesh
        @Takeout (controlpoint_IDs, jacobian, inverse_jacobian, dets, integral_vals, integral_weights, is_occupied) FROM elements
        @Takeout (itp_func_num, itg_func_num, itg_weight, ref_itp_vals) FROM this_space
        elIDs = findall(elements.is_occupied)
        cpIDs = elements.controlpoint_IDs[:, elIDs]
        for X_dim = 1:$dim
            $jac_block
        end
        # integral_vals[:, :, ($ID_for_no_diff)..., elIDs] .= ref_itp_vals[:, :, ($ID_for_no_diff)...]
        integral_vals[:, :, $(ID_for_no_diff...), elIDs] .= ref_itp_vals[:, :, $(ID_for_no_diff...)]
        CUDA.@sync @Dumb_CUDA_Batch 256 $invJac_kernel(jacobian, dets, inverse_jacobian, elIDs)

        CUDA.@sync @Dumb_CUDA_Batch 256 itpval_kernel(integral_vals, ref_itp_vals, inverse_jacobian, elIDs)

        integral_weights .= itg_weight .* dets
    end)
    return ex
end

function gen_BasicBoundary_Funcs(dim::Integer)
    ID_for_no_diff = fill(1, dim)
    jac_block = :(begin
        X_IDs = ntuple(x -> x == X_dim ? 2 : 1, $dim)
    end)
    for this_dim = 1:dim
        x_sym = Symbol(string("x", this_dim))
        push!(jac_block.args, :(jacobian[$this_dim, X_dim, :, facet_IDs] .= bdy_ref_itp_vals[eindex][:, :, X_IDs...] * controlpoints.$x_sym[cpIDs]))
    end

    invJac_kernel = Symbol(string("inv_Jac_", dim, "D"))
    tangent_kernel = Symbol(string("update_Basic_Tangent_", dim, "D"))
    normal_kernel = Symbol(string("update_Basic_Normal_", dim, "D"))
    kernel_libs = Symbol(string("BASE_KERNELS_", dim, "D"))
    func = Symbol(string("update_BasicBoundary_", dim, "D"))
    ex = :(function ($func)(mesh::FEM_WP_Mesh, this_space::Classical_Discretization, itpval_kernel::Function)
        @Takeout (controlpoints, facets, elements) FROM mesh
        @Takeout (tangent_directions, normal_directions, jacobian, inverse_jacobian, el_dets, bdy_dets, 
                  element_ID, element_eindex, integral_vals, integral_weights) FROM facets
        @Takeout (itp_func_num, bdy_itg_func_num, bdy_itg_weights, bdy_tangent_directions, bdy_ref_itp_vals) FROM this_space

        Threads.@threads for eindex = 1:length(bdy_ref_itp_vals)
            facet_IDs = findall(facets.is_occupied .& (element_eindex .== eindex))
            # facet_IDs = findall(facets.is_occupied .& (element_eindex .== eindex) .& (facets.outer_element_ID .== 0))
            isempty(facet_IDs) && continue

            cpIDs = elements.controlpoint_IDs[:, element_ID[facet_IDs]]
            for X_dim = 1:$dim
                $jac_block
            end
            integral_vals[:, :, ($ID_for_no_diff)..., facet_IDs] .= bdy_ref_itp_vals[eindex][:, :, ($ID_for_no_diff)...]
            CUDA.@sync @Dumb_CUDA_Batch 256 $invJac_kernel(jacobian, el_dets, inverse_jacobian, facet_IDs) #Note this dets do not make sense
            CUDA.@sync @Dumb_CUDA_Batch 256 $tangent_kernel(jacobian, tangent_directions, bdy_tangent_directions[eindex], facet_IDs)
            CUDA.@sync @Dumb_CUDA_Batch 256 $normal_kernel(normal_directions, tangent_directions, bdy_dets, facet_IDs)

            CUDA.@sync @Dumb_CUDA_Batch 256 itpval_kernel(integral_vals, bdy_ref_itp_vals[eindex], inverse_jacobian, facet_IDs)

            integral_weights[:, facet_IDs] .= bdy_itg_weights[eindex] .* bdy_dets[:, facet_IDs]
        end
    end)
    return ex
end

@Dumb_Kernel inv_Jac_2D(jacobian, dets, inverse_jacobian, IDs) begin
    this_ID = IDs[thread_idx]
    for itg_id = 1:size(dets, 1)
        dets[itg_id, this_ID] = jacobian[1, 1, itg_id, this_ID] * jacobian[2, 2, itg_id, this_ID] -
                                jacobian[1, 2, itg_id, this_ID] * jacobian[2, 1, itg_id, this_ID]

        inverse_jacobian[1, 1, itg_id, this_ID] =   jacobian[2, 2, itg_id, this_ID] / dets[itg_id, this_ID]
        inverse_jacobian[1, 2, itg_id, this_ID] = - jacobian[1, 2, itg_id, this_ID] / dets[itg_id, this_ID]
        inverse_jacobian[2, 1, itg_id, this_ID] = - jacobian[2, 1, itg_id, this_ID] / dets[itg_id, this_ID]
        inverse_jacobian[2, 2, itg_id, this_ID] =   jacobian[1, 1, itg_id, this_ID] / dets[itg_id, this_ID]
    end
end

@Dumb_Kernel inv_Jac_3D(jacobian, dets, inverse_jacobian, IDs) begin
    this_ID = IDs[thread_idx]
    for itg_id = 1:size(dets, 1)
        dets[itg_id, this_ID] = jacobian[1, 1, itg_id, this_ID] * jacobian[2, 2, itg_id, this_ID] * jacobian[3, 3, itg_id, this_ID] -
                                jacobian[1, 1, itg_id, this_ID] * jacobian[2, 3, itg_id, this_ID] * jacobian[3, 2, itg_id, this_ID] -
                                jacobian[1, 2, itg_id, this_ID] * jacobian[2, 1, itg_id, this_ID] * jacobian[3, 3, itg_id, this_ID] +
                                jacobian[1, 2, itg_id, this_ID] * jacobian[2, 3, itg_id, this_ID] * jacobian[3, 1, itg_id, this_ID] +
                                jacobian[1, 3, itg_id, this_ID] * jacobian[2, 1, itg_id, this_ID] * jacobian[3, 2, itg_id, this_ID] -
                                jacobian[1, 3, itg_id, this_ID] * jacobian[2, 2, itg_id, this_ID] * jacobian[3, 1, itg_id, this_ID]

        inverse_jacobian[1, 1, itg_id, this_ID] = (jacobian[2, 2, itg_id, this_ID] * jacobian[3, 3, itg_id, this_ID] - 
                                                   jacobian[2, 3, itg_id, this_ID] * jacobian[3, 2, itg_id, this_ID]) / dets[itg_id, this_ID]
        inverse_jacobian[1, 2, itg_id, this_ID] = (jacobian[1, 3, itg_id, this_ID] * jacobian[3, 2, itg_id, this_ID] - 
                                                   jacobian[1, 2, itg_id, this_ID] * jacobian[3, 3, itg_id, this_ID]) / dets[itg_id, this_ID]       
        inverse_jacobian[1, 3, itg_id, this_ID] = (jacobian[1, 2, itg_id, this_ID] * jacobian[2, 3, itg_id, this_ID] - 
                                                   jacobian[2, 2, itg_id, this_ID] * jacobian[1, 3, itg_id, this_ID]) / dets[itg_id, this_ID]   

        inverse_jacobian[2, 1, itg_id, this_ID] = (jacobian[2, 3, itg_id, this_ID] * jacobian[3, 1, itg_id, this_ID] - 
                                                   jacobian[3, 3, itg_id, this_ID] * jacobian[2, 1, itg_id, this_ID]) / dets[itg_id, this_ID]
        inverse_jacobian[2, 2, itg_id, this_ID] = (jacobian[1, 1, itg_id, this_ID] * jacobian[3, 3, itg_id, this_ID] - 
                                                   jacobian[1, 3, itg_id, this_ID] * jacobian[3, 1, itg_id, this_ID]) / dets[itg_id, this_ID]       
        inverse_jacobian[2, 3, itg_id, this_ID] = (jacobian[1, 3, itg_id, this_ID] * jacobian[2, 1, itg_id, this_ID] - 
                                                   jacobian[1, 1, itg_id, this_ID] * jacobian[2, 3, itg_id, this_ID]) / dets[itg_id, this_ID]   

        inverse_jacobian[3, 1, itg_id, this_ID] = (jacobian[2, 1, itg_id, this_ID] * jacobian[3, 2, itg_id, this_ID] - 
                                                   jacobian[2, 2, itg_id, this_ID] * jacobian[3, 1, itg_id, this_ID]) / dets[itg_id, this_ID]
        inverse_jacobian[3, 2, itg_id, this_ID] = (jacobian[1, 2, itg_id, this_ID] * jacobian[3, 1, itg_id, this_ID] - 
                                                   jacobian[3, 2, itg_id, this_ID] * jacobian[1, 1, itg_id, this_ID]) / dets[itg_id, this_ID]       
        inverse_jacobian[3, 3, itg_id, this_ID] = (jacobian[1, 1, itg_id, this_ID] * jacobian[2, 2, itg_id, this_ID] - 
                                                   jacobian[2, 1, itg_id, this_ID] * jacobian[1, 2, itg_id, this_ID]) / dets[itg_id, this_ID]                                              
    end
end

#sd_ids: repeatable dimension numbers, e.g., (1,1,1,3)
#sd_IDs: table of diff order .+ 1 for indexing, e.g., (4,1,2) , i.e., 3 times diff on dim 1, no diff on dim 2 and 1 diff on dim 3
sd_ids_To_sd_IDs(dim::Integer, sd_ids) = isempty(sd_ids) ? ntuple(x -> 1, dim) : ntuple(x -> sum(sd_ids .== x) + 1, dim) 
function gen_Kernel_Itpval(max_sd_order::Integer, dim::Integer)
    content = Expr(:block)
    itpval_kernel = gen_Kernel_Name("update_Basic_itgval_", max_sd_order, dim)
    for this_sd_order = 1:max_sd_order
        for sd_ids in Iterators.product(fill(1:dim, this_sd_order)...)
            (length(sd_ids) > 1) && (sum(sd_ids[2:end] .> sd_ids[1:(end - 1)]) > 0) && continue #no repeat
            sd_IDs = sd_ids_To_sd_IDs(dim, sd_ids)
            arg_rhs = Expr(:call, :+)
            for muldim_ids in Iterators.product(fill(1:dim, this_sd_order)...)
                muldim_IDs = sd_ids_To_sd_IDs(dim, muldim_ids)
                single_term = Expr(:call, :*, :(ref_itp_vals[itg_id, itp_id, $(muldim_IDs...)]))

                for (this_sd_id, this_muldim_id) in zip(sd_ids, muldim_ids)
                    push!(single_term.args, :(inverse_jacobian[$this_muldim_id, $this_sd_id, itg_id, this_ID]))
                end
                push!(arg_rhs.args, single_term)
            end
            push!(content.args, :(itg_vals[itg_id, itp_id, $(sd_IDs...), this_ID] = $arg_rhs))
        end
    end
    ex = :(@Dumb_Kernel ($itpval_kernel)(itg_vals, ref_itp_vals, inverse_jacobian, IDs) begin
        this_ID = IDs[thread_idx]
        for itg_id = 1:size(itg_vals, 1)
            for itp_id = 1:size(itg_vals, 2)
                $content
            end
        end
    end)
    return ex
end

BASE_KERNELS_2D = [Core.eval(@__MODULE__, gen_Kernel_Itpval(i, 2)) for i = 1:2]
BASE_KERNELS_3D = [Core.eval(@__MODULE__, gen_Kernel_Itpval(i, 3)) for i = 1:2]
for dim = 2:3
    Core.eval(@__MODULE__, gen_BasicDomain_Funcs(dim))
    Core.eval(@__MODULE__, gen_BasicBoundary_Funcs(dim))
end

@Dumb_Kernel update_Basic_Tangent_2D(jacobian, tangent_directions, bdy_tangent_directions, IDs) begin
    this_ID = IDs[thread_idx]
    for itg_id = 1:size(jacobian, 3)
        tangent_directions[itg_id, 1, 1, this_ID] = jacobian[1, 1, itg_id, this_ID] * bdy_tangent_directions[itg_id, 1, 1] +
                                                    jacobian[1, 2, itg_id, this_ID] * bdy_tangent_directions[itg_id, 2, 1]
        tangent_directions[itg_id, 2, 1, this_ID] = jacobian[2, 1, itg_id, this_ID] * bdy_tangent_directions[itg_id, 1, 1] +
                                                    jacobian[2, 2, itg_id, this_ID] * bdy_tangent_directions[itg_id, 2, 1]
    end
end

@Dumb_Kernel update_Basic_Tangent_3D(jacobian, tangent_directions, bdy_tangent_directions, IDs) begin
    this_ID = IDs[thread_idx]
    for itg_id = 1:size(jacobian, 3)
        tangent_directions[itg_id, 1, 1, this_ID] = jacobian[1, 1, itg_id, this_ID] * bdy_tangent_directions[itg_id, 1, 1] +
                                                    jacobian[1, 2, itg_id, this_ID] * bdy_tangent_directions[itg_id, 2, 1] +
                                                    jacobian[1, 3, itg_id, this_ID] * bdy_tangent_directions[itg_id, 3, 1]
        tangent_directions[itg_id, 2, 1, this_ID] = jacobian[2, 1, itg_id, this_ID] * bdy_tangent_directions[itg_id, 1, 1] +
                                                    jacobian[2, 2, itg_id, this_ID] * bdy_tangent_directions[itg_id, 2, 1] +
                                                    jacobian[2, 3, itg_id, this_ID] * bdy_tangent_directions[itg_id, 3, 1]     
        tangent_directions[itg_id, 3, 1, this_ID] = jacobian[3, 1, itg_id, this_ID] * bdy_tangent_directions[itg_id, 1, 1] +
                                                    jacobian[3, 2, itg_id, this_ID] * bdy_tangent_directions[itg_id, 2, 1] +
                                                    jacobian[3, 3, itg_id, this_ID] * bdy_tangent_directions[itg_id, 3, 1]   

        tangent_directions[itg_id, 1, 2, this_ID] = jacobian[1, 1, itg_id, this_ID] * bdy_tangent_directions[itg_id, 1, 2] +
                                                    jacobian[1, 2, itg_id, this_ID] * bdy_tangent_directions[itg_id, 2, 2] +
                                                    jacobian[1, 3, itg_id, this_ID] * bdy_tangent_directions[itg_id, 3, 2]
        tangent_directions[itg_id, 2, 2, this_ID] = jacobian[2, 1, itg_id, this_ID] * bdy_tangent_directions[itg_id, 1, 2] +
                                                    jacobian[2, 2, itg_id, this_ID] * bdy_tangent_directions[itg_id, 2, 2] +
                                                    jacobian[2, 3, itg_id, this_ID] * bdy_tangent_directions[itg_id, 3, 2]     
        tangent_directions[itg_id, 3, 2, this_ID] = jacobian[3, 1, itg_id, this_ID] * bdy_tangent_directions[itg_id, 1, 2] +
                                                    jacobian[3, 2, itg_id, this_ID] * bdy_tangent_directions[itg_id, 2, 2] +
                                                    jacobian[3, 3, itg_id, this_ID] * bdy_tangent_directions[itg_id, 3, 2] 
    end
end

@Dumb_Kernel update_Basic_Normal_2D(normal_directions, tangent_directions, dets, IDs) begin
    this_ID = IDs[thread_idx]
    for itg_id = 1:size(normal_directions, 1)
        t1, t2 = tangent_directions[itg_id, 1, 1, this_ID], tangent_directions[itg_id, 2, 1, this_ID]
        # dets[itg_id, this_ID] = CUDA.sqrt(CUDA.pow(t1, 2.) + CUDA.pow(t2, 2.))
        dets[itg_id, this_ID] = sqrt(t1 ^ 2. + t2 ^ 2.)
        local_det = dets[itg_id, this_ID]
        normal_directions[itg_id, 1, this_ID] =  t2 / local_det
        normal_directions[itg_id, 2, this_ID] = -t1 / local_det
    end
end

@Dumb_Kernel update_Basic_Normal_3D(normal_directions, tangent_directions, dets, IDs) begin
    this_ID = IDs[thread_idx]
    for itg_id = 1:size(normal_directions, 1)
        t1_1, t2_1, t3_1 = tangent_directions[itg_id, 1, 1, this_ID], tangent_directions[itg_id, 2, 1, this_ID], tangent_directions[itg_id, 3, 1, this_ID]
        t1_2, t2_2, t3_2 = tangent_directions[itg_id, 1, 2, this_ID], tangent_directions[itg_id, 2, 2, this_ID], tangent_directions[itg_id, 3, 2, this_ID]

        rn_1 =   t2_1 * t3_2 - t3_1 * t2_2
        rn_2 = - t1_1 * t3_2 + t3_1 * t1_2
        rn_3 =   t1_1 * t2_2 - t2_1 * t1_2
        # local_det = dets[itg_id, this_ID] = CUDA.sqrt(CUDA.pow(rn_1, 2) + CUDA.pow(rn_2, 2) + CUDA.pow(rn_3, 2))
        local_det = sqrt(rn_1 ^ 2. + rn_2 ^ 2. + rn_3 ^ 2.)
        dets[itg_id, this_ID] = local_det

        normal_directions[itg_id, 1, this_ID] = rn_1 / local_det
        normal_directions[itg_id, 2, this_ID] = rn_2 / local_det
        normal_directions[itg_id, 3, this_ID] = rn_3 / local_det
    end
end
