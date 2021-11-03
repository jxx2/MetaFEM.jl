extract_Words(source::Number) = SymbolicWord[]
extract_Words(source::SymbolicWord) = [source]
extract_Words(source::SymbolicTerm) = union(extract_Words.(source.subterms)...)
extract_Words(source::Symbolic_BilinearForm) = extract_Words(source.base_term)
extract_Words(source::Symbolic_WeakForm) = union(extract_Words.(source.bilinear_forms)...)

function classify_Words(words::Vector{SymbolicWord})
    is_extervar = map(x -> :EXTERNAL_VAR in VARIABLE_ATTRIBUTES[word_To_SymType(x)], words)
    return words[.~is_extervar], words[is_extervar]
end

construct_InnervarInfo(dim::Integer, source::SymbolicWord, bvar_mapping::Dict{Symbol, FEM_Int}) =
    (word_To_TotalSym(dim, source), source.td_order, source.sd_ids, bvar_mapping[word_To_BaseSym(dim, source)])
construct_ExtervarInfo(dim::Integer, source::SymbolicWord) =
    (word_To_TotalSym(dim, source), word_To_LocalSym(dim, source), word_To_SymType(source), source.sd_ids, source.c_ids)

function construct_AssembleTerm(dim::Integer, source::GroundTerm, bvar_mapping::Dict{Symbol, FEM_Int})
    innervar_words, extervar_words = source |> extract_Words |> classify_Words
    ex = parse_Term2Expr(dim, source)
    innervar_infos = construct_InnervarInfo.(dim, innervar_words, Ref(bvar_mapping))
    extervar_infos = construct_ExtervarInfo.(dim, extervar_words)
    return @Construct AssembleTerm
end

function diff_AssembleTerm(dim::Integer, source::GroundTerm, bvar_mapping::Dict{Symbol, FEM_Int})
    innervar_words, extervar_words = source |> extract_Words |> classify_Words
    diff_terms = do_SymbolicDiff.(Ref(source), innervar_words)
    innervar_infos = construct_InnervarInfo.(dim, innervar_words, Ref(bvar_mapping))
    return construct_AssembleTerm.(dim, diff_terms, Ref(bvar_mapping)), innervar_infos
end

function construct_AssembleBilinear(dim::Integer, source::Symbolic_BilinearForm, bvar_mapping::Dict{Symbol, FEM_Int})
    @Takeout (dual_term, base_term) FROM source
    residue_dual_info = construct_InnervarInfo(dim, dual_term, bvar_mapping)
    residue_base_term = construct_AssembleTerm(dim, base_term, bvar_mapping)

    residue_bilinear = AssembleBilinear(residue_dual_info, residue_base_term, tuple(:nothing, 0, (), 0))

    diff_base_terms, innervar_infos = diff_AssembleTerm(dim, base_term, bvar_mapping)
    gradient_bilinears = AssembleBilinear.(Ref(residue_dual_info), diff_base_terms, innervar_infos)
    return residue_bilinear, gradient_bilinears
end

is_Linear(this_bilinear::AssembleBilinear) = isempty(this_bilinear.base_term.innervar_infos)
is_NonLinear(this_bilinear::AssembleBilinear) = ~isempty(this_bilinear.base_term.innervar_infos)

function get_InnerVars(bilinears::Vector{AssembleBilinear})
    vars = getfield.(getfield.(bilinears, :base_term), :innervar_infos)
    return isempty(vars) ? InnervarInfo[] : union(vars...)
end
function get_ExterVars(bilinears::Vector{AssembleBilinear})
    vars = getfield.(getfield.(bilinears, :base_term), :extervar_infos)
    return isempty(vars) ? ExtervarInfo[] : union(vars...)
end

function construct_AssembleWeakform(dim::Integer, source::Symbolic_WeakForm, bvar_mapping::Dict{Symbol, FEM_Int})
    bilinears = construct_AssembleBilinear.(dim, source.bilinear_forms, Ref(bvar_mapping))
    residues = getindex.(bilinears, 1)
    gradients = vcat((getindex.(bilinears, 2))...)

    residueID_by_pos = Dict{FEM_Int, Vector{FEM_Int}}()
    for (this_id, this_bilinear) in pairs(residues)
        _, _, _, base_pos = this_bilinear.dual_info
        ID_box = get(residueID_by_pos, base_pos, FEM_Int[])
        residueID_by_pos[base_pos] = push!(ID_box, this_id)
    end
    gradID_by_pos = Dict{Tuple{FEM_Int, FEM_Int}, Vector{FEM_Int}}()
    for (this_id, this_bilinear) in pairs(gradients)
        _, _, _, dual_pos = this_bilinear.dual_info
        _, _, _, base_pos = this_bilinear.derivative_info
        this_pos = (dual_pos, base_pos)
        ID_box = get(gradID_by_pos, this_pos, FEM_Int[])
        gradID_by_pos[this_pos] = push!(ID_box, this_id)
    end
    return @Construct AssembleWeakform
end
get_SparsePos(wf::AssembleWeakform) = wf.gradID_by_pos |> keys |> collect

function initialize_LocalAssembly(dim::Integer, workpieces::Vector{WorkPiece}; explicit_max_sd_order::Integer = 9)
    for wp in workpieces
        @Takeout (extra_var, domain_weakform, boundary_weakform_pairs) FROM wp.physics

        domain_words = extract_Words(domain_weakform)
        if isempty(boundary_weakform_pairs) #In rare case there may be no boundary
            total_words = domain_words
        else
            boundary_words = union((boundary_weakform_pairs |> values .|> extract_Words)...)
            total_words = union(domain_words, boundary_words)
        end

        sd_orders = (x -> length(x.sd_ids)).(total_words) |> maximum
        max_sd_order = isempty(sd_orders) ? 1 : maximum(sd_orders)

        innervar_words, extervar_words = classify_Words(total_words)        
        workpiece_words = filter(x -> :CONTROLPOINT_VAR in VARIABLE_ATTRIBUTES[word_To_SymType(x)], extervar_words)

        basic_vars = word_To_BaseSym.(dim, innervar_words) |> sort |> unique
        bvar_mapping = Dict(var => FEM_Int(i - 1) for (i, var) in enumerate(basic_vars))

        local_innervar_infos = map(x -> (word_To_LocalSym(dim, x), bvar_mapping[word_To_BaseSym(dim, x)], x.td_order), innervar_words) |> unique
        local_extervars = append!(copy(extra_var), word_To_LocalSym.(dim, workpiece_words))

        assembled_boundary_weakform_pairs = Dict(i => construct_AssembleWeakform(dim, wf, bvar_mapping) for (i, wf) in boundary_weakform_pairs)
        assembled_weakform = construct_AssembleWeakform(dim, domain_weakform, bvar_mapping)

        sparse_entry_ID, sparse_unitsize = (0, 0) .|> FEM_Int
        parse_poses = union(get_SparsePos(assembled_weakform), (assembled_boundary_weakform_pairs |> values |> collect .|> get_SparsePos)...) |> sort
        sparse_mapping = Dict(parse_pose => (i - 1) for (i, parse_pose) in enumerate(sort(parse_poses)))
        # local_asm, max_sd_order = construct_LocalAssembly(wp)
        wp.local_assembly = @Construct FEM_LocalAssembly
        wp.max_sd_order = min(max_sd_order, explicit_max_sd_order)
    end
end
initialize_LocalAssembly(fem_domain::FEM_Domain; explicit_max_sd_order::Integer = 9) = initialize_LocalAssembly(fem_domain.dim, fem_domain.workpieces; explicit_max_sd_order)

get_MaxTimeSteps(local_asm::FEM_LocalAssembly) = getindex.(local_asm.local_innervar_infos, 3) |> maximum
get_MaxTimeSteps(wp::WorkPiece) = get_MaxTimeSteps(wp.local_assembly)
