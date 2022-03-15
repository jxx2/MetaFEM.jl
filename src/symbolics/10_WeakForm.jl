eval_Constant!(x::FEM_Float) = x
function eval_Constant!(x::SymbolicWord) 
    if (x.base_variable in keys(VARIABLE_ATTRIBUTES)) || (x.base_variable in keys(DEFINITION_TABLE)) 
        return x
    else
        local_def = Core.eval(Main, x.base_variable)
        local_def isa Number || error("The \"$local_def\" is not declared before, so it can only be a constant number, but it is not.")
        return FEM_Float(local_def)
    end
end
function eval_Constant!(this_term::SymbolicTerm)
    subterms = this_term.subterms
    for i = 1:length(subterms)
        subterms[i] = eval_Constant!(subterms[i]) 
    end
    return this_term
end

_print_hline() = println("$(repeat("-", 100))")
function FEM_Define!(base_sym::Symbol, term_ex, declared_free_index::Vector{Symbol} = Symbol[])
    this_term = construct_Term(term_ex) |> eval_Constant! |> simplify_Common
    free_index = get_FreeIndex(this_term)
    attributes = get_VarAttribute(base_sym)
    if (:INTEGRATION_POINT_VAR in attributes)
        isempty(free_index) || error("Integration variables are external variables, so the definition must be concrete and has no free index, such as $free_index")
    else
        ((:INTERNAL_VAR in attributes) || (:EXTERNAL_VAR in attributes)) && error("Controlpoint or global variable should not be defined.")
        isempty(symdiff(declared_free_index, free_index)) || error("Free indices must match for readability, but not in $declared_free_index vs $free_index")
    end

    _print_hline()
    if isempty(declared_free_index)
        println("Scalar $base_sym is declared as $(visualize(this_term)).")
    else
        println("Tensor $base_sym{$(join(declared_free_index, ", "))} is declared as $(visualize(this_term))")
    end
    DEFINITION_TABLE[base_sym] = (declared_free_index, this_term)
    return nothing
end
"""
    @Sym b1, c1, b2, c2

    @Def a1 = f(b1, c1) 
    @Def begin
        a2{i} = b2{i} + c2{i}
        a3{i,j} = a2{i} * b2{j}
        ...
    end
`@Def a1 = f(b1, c1)` defines scalar `a` as `f(b, c)` while `a3{i,j} = a2{i} * b2{j}` defines tensor `a3` being the tensor product of vector `a2`,`b2`. Note, free indices are required to match.
"""
macro Def(input_ex)
    input_ex_batch = vectorize_Args(input_ex)
    output_ex = Expr(:block)
    for this_ex in input_ex_batch
        lhs, rhs = this_ex.args
        if lhs isa Symbol
            term_name = lhs
            declared_free_index = Symbol[]
        elseif lhs.head == :curly && length(lhs.args) > 1 && lhs.args[2] isa Symbol
            term_name = lhs.args[1]
            declared_free_index = [x for x in lhs.args[2:end]]
            declared_free_index isa Vector{Symbol} || error("Grammar error, please declare a tensor by whole using kronecker delta instead of directly assigning components. A{1} = 1 is bad, while A{i} = δ{i,1} is good.")
        else
            error("Wrong grammar, lhs = $lhs, rhs = $rhs")
        end
        push!(output_ex.args, :(FEM_Define!($(Meta.quot(term_name)), $(Meta.quot(rhs)), $declared_free_index)))
        push!(output_ex.args, :($term_name = SymbolicWord($(Meta.quot(term_name)))))
    end
    return esc(output_ex)
end

function build_WeakForm(tb::TensorTable, src_def)
    raw_BFs = collect_BilinearTerms!(tb, SymbolicTerm[], src_def isa SymbolicTerm ? unroll_And_Simplify(src_def) : src_def)
    db_dict = regulate_LHS!(tb, raw_BFs)
    return Symbolic_BilinearForm[Symbolic_BilinearForm(this_dual, simplify_Common(⨁(bases))) for (this_dual, bases) in db_dict]
end

collect_BilinearTerms!(tb::TensorTable, buffer_vec::Vector{SymbolicTerm}, ::Number) = buffer_vec
function collect_BilinearTerms!(tb::TensorTable, buffer_vec::Vector{SymbolicTerm}, source_word::SymbolicWord)
    sym_attribute = get_VarAttribute(source_word)
    if ~(:INTERNAL_VAR in sym_attribute) && ~(:EXTERNAL_VAR in sym_attribute)
        if isempty(source_word.sd_ids) && (source_word.td_order == 0) 
            raw_ids, raw_def = deepcopy(DEFINITION_TABLE[source_word.base_variable])
            target_def = substitute_Term!(unroll_And_Simplify(raw_def, tb.dim), raw_ids, source_word.c_ids)
            return collect_BilinearTerms!(tb, buffer_vec, target_def)
        end
    end
    return buffer_vec
end
function collect_BilinearTerms!(tb::TensorTable, buffer_vec::Vector{SymbolicTerm}, source_term::SymbolicTerm)
    term_op = source_term.operation
    if term_op == :Bilinear
        push!(buffer_vec, deepcopy(source_term))
    elseif term_op == :+
        for subterm in source_term.subterms
            collect_BilinearTerms!(tb, buffer_vec, subterm)
        end
    elseif term_op == :*
        sub_vecs = Vector{SymbolicTerm}[collect_BilinearTerms!(tb, SymbolicTerm[], subterm) for subterm in source_term.subterms]
        is_normal = isempty.(sub_vecs)
        bilinear_IDs = findall(.~is_normal)
        if length(bilinear_IDs) > 1 
            error("One product should only contain one BilinearTerm in: $source_term")
        elseif length(bilinear_IDs) == 1 
            other_terms = source_term.subterms[is_normal]
            for bil in sub_vecs[bilinear_IDs[1]]
                new_base_term = ⨂(push!(deepcopy(other_terms), bil.subterms[2]))
                push!(buffer_vec, construct_Term(:Bilinear, [bil.subterms[1], new_base_term]))
            end
        end
    end
    return buffer_vec
end

function regulate_LHS!(tb::TensorTable, raw_BFs::Vector{SymbolicTerm})
    db_dict = Dict{SymbolicWord, Vector{GroundTerm}}()
    for raw_BF in raw_BFs
        raw_dual_term, raw_base_term = raw_BF.subterms
        for (dual_word, factors) in collect_Variations(raw_dual_term, tb)
            push!(get!(db_dict, dual_word, GroundTerm[]), simplify_Common(⨂([⨁(factors), raw_base_term])))
        end
    end
    return db_dict
end