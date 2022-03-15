extract_Words(tb::TensorTable, x) = _extract_Words!(tb, Set{SymbolicWord}(), Set{SymbolicWord}(), x)
_extract_Words!(tb::TensorTable, internal_words::Set{SymbolicWord}, external_words::Set{SymbolicWord}, source::Number) = internal_words, external_words
function _extract_Words!(tb::TensorTable, internal_words::Set{SymbolicWord}, external_words::Set{SymbolicWord}, source::SymbolicWord)
    var_attribute = get_VarAttribute(source)
    if :INTERNAL_VAR in var_attribute
        push!(internal_words, source)
    elseif :EXTERNAL_VAR in var_attribute
        if (:INTEGRATION_POINT_VAR in var_attribute) && (source.base_variable != :n)
            _extract_Words!(tb, internal_words, external_words, DEFINITION_TABLE[source.base_variable][2])
        else
            push!(external_words, source)
        end
    else
        _extract_Words!(tb, internal_words, external_words, evaluate_Tensor(tb, source))
    end
    return internal_words, external_words
end
_extract_Words!(tb::TensorTable, internal_words::Set{SymbolicWord}, external_words::Set{SymbolicWord}, source::SymbolicTerm) = _extract_Words!(tb, internal_words, external_words, source.subterms)
_extract_Words!(tb::TensorTable, internal_words::Set{SymbolicWord}, external_words::Set{SymbolicWord}, source::Symbolic_BilinearForm) = _extract_Words!(tb, internal_words, external_words, source.base_term)
function _extract_Words!(tb::TensorTable, internal_words::Set{SymbolicWord}, external_words::Set{SymbolicWord}, srcs::Vector) 
    for _src in srcs
        _extract_Words!(tb, internal_words, external_words, _src)
    end
    return internal_words, external_words
end

construct_InnervarInfo(dim::Integer, source::SymbolicWord, bvar_mapping::Dict{Symbol, FEM_Int}) = (word_To_TotalSym(dim, source), source.td_order, Tuple(source.sd_ids), bvar_mapping[word_To_BaseSym(dim, source)])
construct_ExtervarInfo(dim::Integer, source::SymbolicWord) = (word_To_TotalSym(dim, source), word_To_LocalSym(dim, source), source.base_variable, Tuple(source.sd_ids), Tuple(source.c_ids))

function construct_AssembleWeakform(tb::TensorTable, srcs::Vector{Symbolic_BilinearForm}, bvar_mapping::Dict{Symbol, FEM_Int})
    dim = tb.dim
    residues, linear_gradients, nonlinear_gradients = AssembleBilinear[], AssembleBilinear[], AssembleBilinear[]
    innervar_infos, linear_extervar_infos, extervar_infos = InnervarInfo[], ExtervarInfo[], ExtervarInfo[]

    for _src in srcs
        @Takeout (dual_word, base_term) FROM _src
        dual_info = construct_InnervarInfo(dim, dual_word, bvar_mapping)
        innervar_words, extervar_words = extract_Words(tb, base_term)

        push!(residues, AssembleBilinear(base_term, dual_info, tuple(:nothing, 0, (), 0)))
        union!(innervar_infos, InnervarInfo[construct_InnervarInfo(dim, word, bvar_mapping) for word in innervar_words])
        union!(extervar_infos, ExtervarInfo[construct_ExtervarInfo(dim, word) for word in extervar_words])
       
        for (diff_word, diffed_termvec) in collect_Variations(base_term, tb)
            diffed_term = simplify_Common(⨁(diffed_termvec))
            diff_innervar_words, diff_extervar_words = extract_Words(tb, diffed_term)
            derivative_info = construct_InnervarInfo(dim, diff_word, bvar_mapping)
            this_diff_bilinear = AssembleBilinear(diffed_term, dual_info, derivative_info)
            if isempty(diff_innervar_words) && prod(Bool[(~(:INTEGRATION_POINT_VAR in get_VarAttribute(word))) || (word.base_variable == :n) for word in diff_extervar_words])
                push!(linear_gradients, this_diff_bilinear)
                union!(linear_extervar_infos, ExtervarInfo[construct_ExtervarInfo(dim, word) for word in diff_extervar_words])
            else
                push!(nonlinear_gradients, this_diff_bilinear)
            end
        end
    end
    return @Construct AssembleWeakform
end

function _collect_SparsePos!(buffer::Set{Tuple{FEM_Int, FEM_Int}}, wf::AssembleWeakform)
    for bil in wf.linear_gradients 
        push!(buffer, (bil.dual_info[4], bil.derivative_info[4]))
    end
    for bil in wf.nonlinear_gradients 
        push!(buffer, (bil.dual_info[4], bil.derivative_info[4]))
    end
    return buffer
end

function collect_SparsePos(wp_wf::AssembleWeakform, bd_wf_pairs::Dict{FEM_Int, AssembleWeakform})
    buffer = _collect_SparsePos!(Set{Tuple{FEM_Int, FEM_Int}}(), wp_wf)
    for (_, wf) in bd_wf_pairs _collect_SparsePos!(buffer, wf) end
    return collect(buffer) |> sort!
end

"""
    initialize_LocalAssembly!(fem_domain::FEM_Domain; explicit_max_sd_order::Integer = 9)
    initialize_LocalAssembly!(tb::TensorTable, workpieces::Vector{WorkPiece}; explicit_max_sd_order::Integer = 9)

This function preprocesses/reorganizes the weakforms. The input `explicit_max_sd_order` is the exposed API for 
explicitly limit high order spatial derivative.
"""
function initialize_LocalAssembly!(tb::TensorTable, workpieces::Vector; explicit_max_sd_order::Integer = 9)
    dim = tb.dim
    for wp in workpieces
        @Takeout (extra_var, domain_weakform, boundary_weakform_pairs) FROM wp.physics

        innervar_words, extervar_words = extract_Words(tb, domain_weakform)
        for (_, wf) in boundary_weakform_pairs
            _extract_Words!(tb, innervar_words, extervar_words, wf)
        end
   
        basic_vars = Symbol[word_To_BaseSym(dim, x) for x in innervar_words] |> sort! |> unique!
        bvar_mapping = Dict(var => FEM_Int(i - 1) for (i, var) in enumerate(basic_vars))

        local_innervar_infos = [(word_To_LocalSym(dim, x), bvar_mapping[word_To_BaseSym(dim, x)], x.td_order) for x in innervar_words] |> unique!
        controlpoint_extervars = [extra_var; word_To_LocalSym.(dim, filter!(x -> :CONTROLPOINT_VAR in get_VarAttribute(x), extervar_words))]

        assembled_boundary_weakform_pairs = Dict(i => construct_AssembleWeakform(tb, wf, bvar_mapping) for (i, wf) in boundary_weakform_pairs)
        assembled_weakform = construct_AssembleWeakform(tb, domain_weakform, bvar_mapping)

        sparse_entry_ID, sparse_unitsize = 0, 0

        parse_poses = collect_SparsePos(assembled_weakform, assembled_boundary_weakform_pairs)
        sparse_mapping = Dict(parse_pose => (i - 1) for (i, parse_pose) in enumerate(parse_poses))

        wp.local_assembly = @Construct FEM_LocalAssembly
        wp.max_sd_order = min(max(collect_SDOrder(assembled_weakform), collect_SDOrder(values(assembled_boundary_weakform_pairs))), explicit_max_sd_order)
        # wp.max_sd_order = min(max(get_MaxSDOrder(innervar_words), get_MaxSDOrder(extervar_words)), explicit_max_sd_order)
    end
end
initialize_LocalAssembly!(fem_domain::FEM_Domain; explicit_max_sd_order::Integer = 9) = initialize_LocalAssembly!(fem_domain.tensor_table, fem_domain.workpieces; explicit_max_sd_order)

collect_SDOrder(x::InnervarInfo) = length(x[3])
collect_SDOrder(x::ExtervarInfo) = length(x[4])
collect_SDOrder(x::AssembleBilinear) = collect_SDOrder(x.dual_info)
collect_SDOrder(x::AssembleWeakform) = max(collect_SDOrder(x.residues), collect_SDOrder(x.innervar_infos), collect_SDOrder(x.extervar_infos))
collect_SDOrder(x) = isempty(x) ? 1 : maximum(collect_SDOrder.(x))

# get_MaxSDOrder(words) = isempty(words) ? 1 : maximum([length(x.sd_ids) for x in words])
get_MaxTimeSteps(local_asm::FEM_LocalAssembly) = maximum([x[3] for x in local_asm.local_innervar_infos])
get_MaxTimeSteps(wp::WorkPiece) = get_MaxTimeSteps(wp.local_assembly)
