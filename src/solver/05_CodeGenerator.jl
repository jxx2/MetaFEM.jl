function declare_Innervar_GPU(dim::Integer, innervar_infos::Vector{InnervarInfo}, max_sd_order::Integer; itg_val_fixed::Bool) 
    declare_code = Expr(:block)
    for (total_sym, td_order, sd_order, basic_pos) in sort(innervar_infos)
        push!(declare_code.args, :($total_sym = FEM_buffer(ArrayType, FEM_Float, local_itg_func_num, elnum)))

        length(sd_order) > max_sd_order && continue
        sd_IDs = sd_ids_To_sd_IDs(dim, sd_order)
        push!(declare_code.args, itg_val_fixed ? 
            :(_Var_Cut(local_integral_vals, $sd_IDs, ($td_order * basicfield_size + $basic_pos * variable_size), global_cpIDs, x_star, $total_sym, elIDs)) : 
            :(_Var_Basic(local_integral_vals, $sd_IDs, ($td_order * basicfield_size + $basic_pos * variable_size), global_cpIDs, x_star, $total_sym, local_itg_hostIDs, elIDs)))
    end
    return declare_code
end

function declare_Extervar_GPU(dim::Integer, extervar_infos::Vector{ExtervarInfo}, max_sd_order::Integer; itg_val_fixed::Bool, is_boundary::Bool = false)
    var_host = is_boundary ? :facets : :elements
    declare_code = Expr(:block)
    for (total_sym, local_sym, type_sym, sd_order, c_ids) in sort(extervar_infos)
        var_attribute = get_VarAttribute(type_sym)
        if :GLOBAL_VAR in var_attribute
            if type_sym == :t
                arg = :($total_sym = globalfield.t)
            elseif type_sym == :dt
                arg = :($total_sym = globalfield.dt)
            else
                arg = :($total_sym = wp.physics.global_vars[$(Meta.quot(total_sym))])
            end
        elseif :CONTROLPOINT_VAR in var_attribute
            push!(declare_code.args, :($total_sym = FEM_buffer(ArrayType, FEM_Float, local_itg_func_num, elnum)))

            length(sd_order) > max_sd_order && continue
            sd_IDs = sd_ids_To_sd_IDs(dim, sd_order)            
            arg = itg_val_fixed ? :(_Var_Cut(local_integral_vals, $sd_IDs, 0, elements.controlpoint_IDs, controlpoints.$local_sym, $total_sym, elIDs)) : 
                                  :(_Var_Basic(local_integral_vals, $sd_IDs, 0, elements.controlpoint_IDs, controlpoints.$local_sym, $total_sym, local_itg_hostIDs, elIDs))
        elseif :INTEGRATION_POINT_VAR in var_attribute
            isempty(sd_order) || error("Integration point variable cant have spatial derivative, use controlpoint variable instead")
            if type_sym == :n 
                length(c_ids) == 1 || error("normal only has one direction")
                is_boundary || error("Body don't have normal")
                arg = :($total_sym = facets.normal_directions[:, $(c_ids[1]), facet_IDs])
            else
                error("Unresolved external variable")
            end
        else
            error("Unresolved external variable")
        end
        push!(declare_code.args, arg)
    end
    return declare_code
end

function gen_K_Linear_GPU(tb::TensorTable, asm_wf::AssembleWeakform, sparse_mapping::Dict{Tuple{FEM_Int, FEM_Int}, FEM_Int}, max_sd_order::Integer; 
    is_boundary::Bool = false, itg_val_fixed::Bool, itg_weight_fixed::Bool) 
    dim = tb.dim
    @Takeout (linear_gradients, linear_extervar_infos) FROM asm_wf
    extervar_declaration = declare_Extervar_GPU(dim, linear_extervar_infos, max_sd_order; itg_val_fixed = itg_val_fixed, is_boundary = is_boundary)

    K_func_code = Expr(:block)
    intermediate_code, declared_syms = Expr[], Set{Symbol}()
    for this_bilinear in linear_gradients
        func_exs = parse_Term2Expr!(intermediate_code, declared_syms, tb, this_bilinear.base_term)
        _, _, dual_sd_order, dual_basic_pos = this_bilinear.dual_info
        _, derivative_td_order, derivative_sd_order, derivative_basic_pos = this_bilinear.derivative_info
        sparse_unit_num = sparse_mapping[(dual_basic_pos, derivative_basic_pos)]

        max(length(dual_sd_order), length(derivative_sd_order)) > max_sd_order && continue
        dual_sd_IDs = sd_ids_To_sd_IDs(dim, dual_sd_order)
        base_sd_IDs = sd_ids_To_sd_IDs(dim, derivative_sd_order)
        
        local_func_code = quote
            sparse_ID_shift = $sparse_unit_num * sparse_unitsize
        end

        for this_func_ex in func_exs
            val_arg = itg_weight_fixed ? :(vals = @. $this_func_ex * K_params[$(derivative_td_order + 1)] * local_integral_weights) : 
                                         :(vals = @. $this_func_ex * K_params[$(derivative_td_order + 1)] * local_integral_weights[:, local_itg_hostIDs])
            ker_arg = itg_val_fixed ? :(_Kval_Cut(local_integral_vals, $dual_sd_IDs, $base_sd_IDs, vals, sparse_IDs_by_el, sparse_ID_shift, K_linear, elIDs)) :
                                      :(_Kval_Basic(local_integral_vals, $dual_sd_IDs, $base_sd_IDs, vals, sparse_IDs_by_el, sparse_ID_shift, K_linear, local_itg_hostIDs, elIDs))
            push!(local_func_code.args, val_arg)
            push!(local_func_code.args, ker_arg)
        end
        push!(K_func_code.args, local_func_code)
    end
    intermediate_declaration = Expr(:block, intermediate_code...)
    final_block = quote
        $extervar_declaration
        $intermediate_declaration
        $K_func_code
    end
    return final_block 
end

function gen_Res_K_NonLinear_GPU(tb::TensorTable, asm_wf::AssembleWeakform, sparse_mapping::Dict{Tuple{FEM_Int, FEM_Int}, FEM_Int}, max_sd_order::Integer;
    is_boundary::Bool = false, itg_val_fixed::Bool, itg_weight_fixed::Bool)
    dim = tb.dim
    @Takeout (residues, nonlinear_gradients, innervar_infos, extervar_infos) FROM asm_wf
    innervar_declaration = declare_Innervar_GPU(dim, innervar_infos, max_sd_order; itg_val_fixed = itg_val_fixed)
    extervar_declaration = declare_Extervar_GPU(dim, extervar_infos, max_sd_order; itg_val_fixed = itg_val_fixed, is_boundary = is_boundary)

    res_func_code = Expr(:block)
    intermediate_code, declared_syms = Expr[], Set{Symbol}()
    for this_bilinear in asm_wf.residues
        func_exs = parse_Term2Expr!(intermediate_code, declared_syms, tb, this_bilinear.base_term)
        _, _, dual_sd_order, dual_basic_pos = this_bilinear.dual_info
        length(dual_sd_order) > max_sd_order && continue
        dual_sd_IDs = sd_ids_To_sd_IDs(dim, dual_sd_order)

        local_func_code = Expr(:block)
        for this_func_ex in func_exs
            val_arg = itg_weight_fixed ? :(vals = @. $this_func_ex * local_integral_weights) : 
                                         :(vals = @. $this_func_ex * local_integral_weights[:, local_itg_hostIDs])
            ker_arg = itg_val_fixed ? :(_Res_Cut(local_integral_vals, $dual_sd_IDs, vals, $dual_basic_pos * variable_size, global_cpIDs, residue, elIDs)) :
                                      :(_Res_Basic(local_integral_vals, $dual_sd_IDs, vals, $dual_basic_pos * variable_size, global_cpIDs, residue, local_itg_hostIDs, elIDs))
            push!(local_func_code.args, val_arg)
            push!(local_func_code.args, ker_arg)
        end
        push!(res_func_code.args, local_func_code)
    end

    K_func_code = Expr(:block)
    for this_bilinear in nonlinear_gradients
        func_exs = parse_Term2Expr!(intermediate_code, declared_syms, tb, this_bilinear.base_term)
        _, _, dual_sd_order, dual_basic_pos = this_bilinear.dual_info
        _, derivative_td_order, derivative_sd_order, derivative_basic_pos = this_bilinear.derivative_info
        sparse_unit_num = sparse_mapping[(dual_basic_pos, derivative_basic_pos)]

        max(length(dual_sd_order), length(derivative_sd_order)) > max_sd_order && continue
        dual_sd_IDs = sd_ids_To_sd_IDs(dim, dual_sd_order)
        base_sd_IDs = sd_ids_To_sd_IDs(dim, derivative_sd_order)

        local_func_code = quote
            sparse_ID_shift = $sparse_unit_num * sparse_unitsize
        end

        for this_func_ex in func_exs
            val_arg = itg_weight_fixed ? :(vals = @. $this_func_ex * K_params[$(derivative_td_order + 1)] * local_integral_weights) : 
                                         :(vals = @. $this_func_ex * K_params[$(derivative_td_order + 1)] * local_integral_weights[:, local_itg_hostIDs])
            ker_arg = itg_val_fixed ? :(_Kval_Cut(local_integral_vals, $dual_sd_IDs, $base_sd_IDs, vals, sparse_IDs_by_el, sparse_ID_shift, K_total, elIDs)) :
                                      :(_Kval_Basic(local_integral_vals, $dual_sd_IDs, $base_sd_IDs, vals, sparse_IDs_by_el, sparse_ID_shift, K_total, local_itg_hostIDs, elIDs))
            push!(local_func_code.args, val_arg)
            push!(local_func_code.args, ker_arg)
        end
        push!(K_func_code.args, local_func_code)
    end
    intermediate_declaration = Expr(:block, intermediate_code...)
    final_block = quote
        $innervar_declaration
        $extervar_declaration
        $intermediate_declaration
        $res_func_code
        $K_func_code
    end
    return final_block 
end

function gen_CodeBody(fem_genfunc::Function; fem_domain::FEM_Domain)
    @Takeout (tensor_table, workpieces, globalfield) FROM fem_domain

    wp_total_code = Expr(:block)
    for (wp_ID, wp) in pairs(workpieces)
        @Takeout (max_sd_order, local_assembly) FROM wp
        @Takeout (assembled_boundary_weakform_pairs, assembled_weakform, sparse_mapping) FROM local_assembly
        if wp.element_space isa Classical_Discretization
            el_block = fem_genfunc(tensor_table, assembled_weakform, sparse_mapping, max_sd_order; is_boundary = false, itg_val_fixed = false, itg_weight_fixed = false)
            el_loop = quote
                elIDs = findall(elements.is_occupied)
                elnum = length(elIDs)
                local_itg_hostIDs = elIDs
                local_integral_vals = elements.integral_vals
                local_integral_weights = elements.integral_weights
                local_itg_func_num = size(local_integral_weights)[1]
                $el_block
            end
            bg_total_loop = quote
                local_integral_vals = facets.integral_vals
                local_integral_weights = facets.integral_weights
                local_itg_func_num = size(local_integral_weights)[1]
            end
            for (bg_ID, bg_asm_wf) in assembled_boundary_weakform_pairs
                bg_local_block = fem_genfunc(tensor_table, bg_asm_wf, sparse_mapping, max_sd_order; is_boundary = true, itg_val_fixed = false, itg_weight_fixed = false)
                bg_loop = quote
                    facet_IDs = bg_fIDs[$bg_ID]
                    elIDs = facets.element_ID[facet_IDs]
                    elnum = length(elIDs)
                    local_itg_hostIDs = facet_IDs
                    $bg_local_block
                end
                push!(bg_total_loop.args, bg_loop)
            end
            wp_local_code = quote
                wp = workpieces[$wp_ID]
                @Takeout (controlpoints, facets, elements, bg_fIDs, variable_size) FROM wp.mesh
                @Takeout sparse_unitsize FROM wp.local_assembly
                @Takeout (global_cpIDs, sparse_IDs_by_el) FROM elements
                $el_loop
                $bg_total_loop
            end
        elseif wp.element_space isa CutCell_Discretization
            inner_el_block = fem_genfunc(tensor_table, assembled_weakform, sparse_mapping, max_sd_order; is_boundary = false, itg_val_fixed = true, itg_weight_fixed = true)
            inner_el_loop = quote
                elIDs = findall(elements.is_occupied .& elements.is_inner)
                if ~isempty(elIDs)
                    elnum = length(elIDs)
                    # local_itg_hostIDs = findall(CUDA.ones(Bool, elnum))
                    local_integral_vals = fem_domain.element_space.integral_vals 
                    local_integral_weights = fem_domain.element_space.integral_weights .* CUDA.ones(FEM_Float, 1, elnum)
                    local_itg_func_num = size(local_integral_weights)[1]
                    $inner_el_block
                end
            end

            bdy_el_block = fem_genfunc(tensor_table, assembled_weakform, sparse_mapping, max_sd_order; is_boundary = false, itg_val_fixed = true, itg_weight_fixed = false)
            bdy_el_loop = quote
                elIDs = findall(elements.is_occupied .& (.~ elements.is_inner))
                if ~isempty(elIDs)
                    elnum = length(elIDs)
                    local_itg_hostIDs = elIDs
                    local_integral_vals = fem_domain.element_space.frag_integral_vals 
                    local_integral_weights = elements.integral_weights
                    local_itg_func_num = size(local_integral_weights)[1]
                    $bdy_el_block
                end
            end

            bg_total_loop = quote
                local_integral_vals = facets.integral_vals
                local_integral_weights = facets.integral_weights
                local_itg_func_num = size(local_integral_weights)[1]
                #local_normals = edges.normal_directions
            end
            for (bg_ID, bg_asm_wf) in assembled_boundary_weakform_pairs
                bg_local_block = fem_genfunc(tensor_table, bg_asm_wf, sparse_mapping, max_sd_order; is_boundary = true, itg_val_fixed = false, itg_weight_fixed = false)
                bg_loop = quote
                    facet_IDs = bg_fIDs[$bg_ID]
                    if ~isempty(facet_IDs)
                        elIDs = facets.element_ID[facet_IDs]
                        elnum = length(elIDs)
                        local_itg_hostIDs = facet_IDs
                        $bg_local_block
                    end
                end
                push!(bg_total_loop.args, bg_loop)
            end

            wp_local_code = quote
                wp = workpieces[$wp_ID]
                @Takeout (controlpoints, facets, elements, bg_fIDs, variable_size) FROM wp.mesh
                @Takeout sparse_unitsize FROM wp.local_assembly
                @Takeout (global_cpIDs, sparse_IDs_by_el) FROM elements
                $inner_el_loop
                $bdy_el_loop
                $bg_total_loop
            end
        end
        push!(wp_total_code.args, wp_local_code)
    end
    return wp_total_code 
end
"""
    compile_Updater_GPU(; domain_ID::Integer, fem_domain::FEM_Domain)

The function generates `fem_domain`.`K_linear_func` and `fem_domain`.`K_nonlinear_func`.
The input `domain_ID` is only used to generate the function name.
"""
function compile_Updater_GPU(; domain_ID::Integer, fem_domain::FEM_Domain)  
    linear_func_name = Symbol("update_K_Linear_", domain_ID)
    linear_func_body = :(
    function ($linear_func_name)(time_discretization::GeneralAlpha; fem_domain::FEM_Domain{ArrayType}) where {ArrayType}
        @Takeout (workpieces, globalfield) FROM fem_domain
        @Takeout (basicfield_size, K_linear) FROM fem_domain.globalfield
        @Takeout K_params FROM time_discretization
        K_linear .= 0.
        $(gen_CodeBody(gen_K_Linear_GPU; fem_domain = fem_domain))
    end)

    nonlinear_func_name = Symbol("update_K_NonLinear_", domain_ID)
    nonlinear_func_body = :(
    function ($nonlinear_func_name)(time_discretization::GeneralAlpha; fem_domain::FEM_Domain{ArrayType}) where {ArrayType}
        @Takeout (workpieces, globalfield) FROM fem_domain
        @Takeout (basicfield_size, x_star, residue, K_linear, K_total) FROM globalfield
        @Takeout K_params FROM time_discretization
        residue .= 0.
        K_total .= K_linear
        $(gen_CodeBody(gen_Res_K_NonLinear_GPU; fem_domain = fem_domain))
    end)

    fem_domain.K_linear_func = eval(linear_func_body)
    fem_domain.K_nonlinear_func = eval(nonlinear_func_body)

    return (linear_func_body, nonlinear_func_body) .|> striplines .|> stripblocks
end
