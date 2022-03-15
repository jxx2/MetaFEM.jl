using Base: isoperator
base_TensorInfo(sym::Symbol, order::Integer) = (sym, order, 0, 0) |> TensorInfo
word_To_TensorInfo(x::SymbolicWord) = (x.base_variable, length(x.c_ids), x.td_order, length(x.sd_ids)) |> TensorInfo
tensorinfo_To_Word(x::TensorInfo, ids::Vector{IndexSym}) = SymbolicWord(x[1], x[3], ids[1:x[2]], ids[(x[2]+ 1):x[4]])
_tensor_DOF(x::TensorInfo) = x[2] + x[4]

unroll_And_Simplify(source_term::GroundTerm, dim::Integer) = unroll_Dumb_Indices(source_term, dim) |> replace_SpecialTerm!
function construct_Tensor(tensor_info::TensorInfo, declared_free_index::Vector{Symbol}, definition::GroundTerm)
    src_free_index = get_FreeIndex(definition)
    isempty(symdiff(declared_free_index, src_free_index)) || error("Free indices must match for readability, but not in $declared_free_index vs $src_free_index in $definition")
    free_index, definition = reindex_Term!(definition, declared_free_index)
    indexed_instances = Dict{Vector, GroundTerm}()
    println("Building $(_tensor_DOF(tensor_info) == 0 ? :Scalar : :Tensor) $tensor_info")
    println(definition)
    if length(free_index) == 2
        swapped_def = substitute_Term!(deepcopy(definition), free_index, reverse(free_index))
        if definition == swapped_def
            println("$(tensor_info[1]) is a symmetric tensor")
            push!(get!(VARIABLE_ATTRIBUTES, tensor_info[1], Symbol[]), :SYMMETRIC_TENSOR)
        end
    end
    return PhysicalTensor(tensor_info, definition, free_index, indexed_instances)
end

function get_Tensor(tb::TensorTable, tensor_info::TensorInfo) 
    if tensor_info in keys(tb.tensors)
        return tb.tensors[tensor_info]
    else
        return tb.tensors[tensor_info] = build_Tensor(tb, tensor_info)
    end
end

function build_Tensor(tb::TensorTable, tensor_info::TensorInfo)
    sym, base_order, td_order, sd_order = tensor_info
    if td_order > 0
        base_tensor = get_Tensor(tb, (sym, base_order, td_order - 1, sd_order) |> TensorInfo)
        target_def = diff_Time(base_tensor.definition) # time derivative doesn't need args
        target_ids = base_tensor.free_index
    elseif sd_order > 0
        base_tensor = get_Tensor(tb, (sym, base_order, td_order, sd_order - 1) |> TensorInfo)
        placeholder = gensym()
        target_def = diff_Space(base_tensor.definition, placeholder)
        target_ids, target_def = reindex_Term!(target_def, [base_tensor.free_index; placeholder])
    else
        target_ids, raw_def = deepcopy(DEFINITION_TABLE[sym])
        target_def = inline_TensorDiff!(tb, unroll_And_Simplify(raw_def, tb.dim))
    end
    return construct_Tensor(tensor_info, target_ids, target_def)
end

collect_ids(this_word::SymbolicWord) = IndexSym[this_word.c_ids; this_word.sd_ids]
function evaluate_Tensor(tb::TensorTable, this_word::SymbolicWord)
    tensor = get_Tensor(tb, word_To_TensorInfo(this_word))
    target_ids = collect_ids(this_word)
    if isempty(target_ids) || target_ids == tensor.free_index
        return tensor.definition
    elseif target_ids in keys(tensor.indexed_instances)
        return tensor.indexed_instances[target_ids]
    else # Here the inline_TensorDiff is to inline the number only
        return tensor.indexed_instances[target_ids] = inline_TensorDiff!(tb, replace_SpecialTerm!(substitute_Term!(deepcopy(tensor.definition), tensor.free_index, target_ids)))
    end
end 

inline_TensorDiff!(tb, x) = _inline_TensorDiff!(tb, x)[2] |> simplify_Common
_inline_TensorDiff!(tb::TensorTable, x::Number) = false, x
function _inline_TensorDiff!(tb::TensorTable, x::SymbolicWord) 
    var_attribute = get_VarAttribute(x)
    if (:INTERNAL_VAR in var_attribute) || (:EXTERNAL_VAR in var_attribute) 
        return false, x
    else
        local_def = evaluate_Tensor(tb, x) 
        if local_def isa Number
            return true, local_def
        else
            return false, x
        end
    end
end

function _inline_TensorDiff!(tb::TensorTable, this_term::SymbolicTerm)
    if this_term.operation == :d
        ((length(this_term.subterms) == 2) && (this_term.subterms[2] isa SymbolicWord)) || error("AST error")
        return true, diff_Symbol(inline_TensorDiff!(tb, this_term.subterms[1]), tb, this_term.subterms[2])
    else
        term_changed, subterms = false, this_term.subterms
        for i = 1:length(subterms)
            subterm_changed, subterms[i] = _inline_TensorDiff!(tb, subterms[i]) 
            term_changed |= subterm_changed
        end
        return term_changed, refresh_Term(this_term, term_changed)
    end
end

function get_TensorDiff!(tb::TensorTable, src_info::TensorInfo, diff_info::TensorInfo) 
    info_pair = (src_info, diff_info)
    if info_pair in keys(tb.diff_tensors)
        return tb.diff_tensors[info_pair]
    else
        return tb.diff_tensors[info_pair] = construct_TensorDiff!(tb, src_info, diff_info)
    end
end

function construct_TensorDiff!(tb::TensorTable, src_info::TensorInfo, diff_info::TensorInfo)
    diff_DOF = _tensor_DOF(diff_info)
    target_info = base_TensorInfo(get_DiffSym(src_info, diff_info), _tensor_DOF(src_info) + diff_DOF)

    src_tensor = get_Tensor(tb, src_info)

    diff_ids = IndexSym[gensym() for i = 1:diff_DOF]
    diff_word = tensorinfo_To_Word(diff_info, diff_ids)

    target_ids = Symbol[src_tensor.free_index; diff_ids]
    target_def = diff_Symbol(src_tensor.definition, tb, diff_word) 
    
    println("Building tensor derivative $target_info by $target_def")
    tb.tensors[target_info] = construct_Tensor(target_info, target_ids, target_def)
    return target_info
end

count_Words(x::Number) = 0
count_Words(x::SymbolicWord) = 1 
count_Words(x::SymbolicTerm) = sum(count_Words.(x.subterms))

propagate_Symbol(tb, x) = _propagate_Symbol!(tb, x)[2] |> simplify_Common
propagate_Symbol(tb, x::SymbolicTerm) = _propagate_Symbol!(tb, deepcopy(x))[2] |> simplify_Common
_propagate_Symbol!(tb::TensorTable, x::Number) = false, x
function _propagate_Symbol!(tb::TensorTable, x::SymbolicWord) 
    var_attribute = get_VarAttribute(x)
    if (:INTERNAL_VAR in var_attribute) || (:EXTERNAL_VAR in var_attribute) 
        return false, x
    else
        local_def = evaluate_Tensor(tb, x) 
        if local_def isa SymbolicTerm
            return (count_Words(local_def) > 1) ? (false, x) : (true, propagate_Symbol(tb, local_def))
        else
            return true, propagate_Symbol(tb, local_def)
        end
    end
end
function _propagate_Symbol!(tb::TensorTable, this_term::SymbolicTerm)
    term_changed, subterms = false, this_term.subterms
    for i = 1:length(subterms)
        subterm_changed, subterms[i] = _propagate_Symbol!(tb, subterms[i]) 
        term_changed |= subterm_changed
    end
    return term_changed, refresh_Term(this_term, term_changed)
end

function generates_All_Related_ITG_Symbols(tb::TensorTable, this_word::SymbolicWord, attributes::Vector{Symbol})
    dim = tb.dim
    ((this_word.td_order == 0) && isempty(this_word.sd_ids)) || error("Integration point can only have direction IDs")
    tensor_order = length(this_word.c_ids)

    if (tensor_order == 0)
        return Symbol[word_To_TotalSym(tb.dim, this_word)]
    elseif (tensor_order == 1)
        return Symbol[word_To_TotalSym(tb.dim, SymbolicWord(this_word.base_variable, 0, IndexSym[FEM_Int(i)], Symbol[])) for i = 1:dim]
    elseif (tensor_order == 2)
        if (:SYMMETRIC_TENSOR in get_VarAttribute(this_word.base_variable))
            return Symbol[word_To_TotalSym(tb.dim, SymbolicWord(this_word.base_variable, 0, IndexSym[inverse_Voigt_ID(i, dim)...], Symbol[])) for i = 1:6]
        else
            return Symbol[word_To_TotalSym(tb.dim, SymbolicWord(this_word.base_variable, 0, IndexSym[FEM_Int(i), FEM_Int(j)], Symbol[])) for i = 1:dim, j = 1:dim]
        end
    else
        error("Order > 3 integration point variables are explicitly forbidden")
    end
end

_parse_Term2Expr!(intermediate_code::Vector{Expr}, declared_syms::Set{Symbol}, tb::TensorTable, this_num::FEM_Float) = this_num
function _parse_Term2Expr!(intermediate_code::Vector{Expr}, declared_syms::Set{Symbol}, tb::TensorTable, this_word::SymbolicWord) 
    totalsym = word_To_TotalSym(tb.dim, this_word)
    if ~(totalsym in declared_syms)
        attributes = get_VarAttribute(this_word)
        if (:INTERNAL_VAR in attributes) || (:EXTERNAL_VAR in attributes)
            if (:INTEGRATION_POINT_VAR in attributes) && (this_word.base_variable != :n)
                all_syms = generates_All_Related_ITG_Symbols(tb, this_word, attributes)
                sym_tuple = (length(all_syms) == 1) ? all_syms[1] : :(($(all_syms...),))

                _, raw_def = deepcopy(DEFINITION_TABLE[this_word.base_variable])
                this_def = _parse_Term2Expr!(intermediate_code, declared_syms, tb, propagate_Symbol(tb, raw_def))
                
                push!(intermediate_code, :($sym_tuple = $this_def))
                union!(declared_syms, all_syms)
            else
                # println("$(this_word.base_variable) is a base_variable")
                push!(declared_syms, totalsym)
            end
        else
            defs = parse_Term2Expr!(intermediate_code, declared_syms, tb, evaluate_Tensor(tb, this_word))
            if defs[1] isa Expr
                push!(intermediate_code, :($totalsym = @. $(defs[1])))
                for this_def in defs[2:end]
                    push!(intermediate_code, :(@. $totalsym += $this_def))
                end
            else
                push!(intermediate_code, :($totalsym = $this_def))
            end
            push!(declared_syms, totalsym)
        end
    end
    return totalsym
end

function _parse_Term2Expr!(intermediate_code::Vector{Expr}, declared_syms::Set{Symbol}, tb::TensorTable, this_term::SymbolicTerm) 
    op = this_term.operation
    args = [_parse_Term2Expr!(intermediate_code, declared_syms, tb, subterm) for subterm in this_term.subterms]
    if isoperator(op) 
        return Expr(:call, op, args...)
    else
        return :(Main.$op($(args...)))
    end
end

function parse_Term2Expr!(intermediate_code::Vector{Expr}, declared_syms::Set{Symbol}, tb::TensorTable, this_term) 
    inlined_term = propagate_Symbol(tb, this_term)

    if inlined_term isa SymbolicTerm && inlined_term.operation == :+
        word_nums = count_Words.(inlined_term.subterms)
        limit, counter = 64, 0
        buffer, defs = GroundTerm[], Expr[]
        for (id, word_num) in enumerate(word_nums)
            if (counter += word_num) > limit
                counter = 0
                push!(defs, _parse_Term2Expr!(intermediate_code, declared_syms, tb, ⨁(buffer)))
                empty!(buffer)
            end
            push!(buffer, inlined_term.subterms[id])
        end
        return push!(defs, _parse_Term2Expr!(intermediate_code, declared_syms, tb, ⨁(buffer))) 
    else 
        return [_parse_Term2Expr!(intermediate_code, declared_syms, tb, inlined_term)]
    end
end