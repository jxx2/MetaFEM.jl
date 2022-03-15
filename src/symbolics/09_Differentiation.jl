# for derivative
@Define_Semantic_Constraint N₁ ∊ (N₁ isa Number)

@Define_Rewrite_Rule Num_Diff ≔ ∂(N₁) => 0

@Define_Rewrite_Rule Add_Diff ≔ ∂(a + (b...)) => ∂(a) + ∂(+(b...))
@Define_Rewrite_Rule Mul_Diff ≔ ∂(a * (b...)) => ∂(a) * (b...) + a * ∂(*(b...))
@Define_Rewrite_Rule Pow_Diff ≔ ∂(a ^ b) => ∂(a) * a ^ (b - 1) * b  + ∂(b) * log(a) * a ^ b
@Define_Rewrite_Rule Log_Diff ≔ ∂(log(a)) => ∂(a) * a ^ (-1) 

@Define_Rewrite_Rule Cond_Diff ≔ ∂(ifelse(a, b, c)) => ifelse(a, ∂(b), ∂(c))

DIFF_RULES = [Add_Diff, Mul_Diff, Pow_Diff, Log_Diff, Cond_Diff, Num_Diff]

function _diff_Core!(source_term)
    this_term, is_changed = construct_Term(:∂, [source_term]), true
    while is_changed
        is_changed, this_term = apply_Rules(this_term, DIFF_RULES)
    end
    return this_term
end

for (tag, args) in zip([:Time, :Space, :Variation, :Symbol], [[], [:(d_id::Symbol)], [:(tb::TensorTable)], [:(tb::TensorTable), :(diff_word::SymbolicWord)]])
    diff_funcname = Symbol("diff_$(tag)")
    eval_global_funcname = Symbol("diff_Eval_$(tag)_Global!")
    eval_local_funcname = Symbol("diff_Eval_$(tag)_Local!")
    @eval begin
        $diff_funcname(x::Number, $(args...)) = 0.
        $diff_funcname(x::Union{SymbolicWord, SymbolicTerm}, $(args...)) = $eval_global_funcname(_diff_Core!(deepcopy(x)), $(args...))[2] |> simplify_Common
        $eval_global_funcname(x::Union{Number, SymbolicWord}, $(args...)) = false, x
        function $(eval_global_funcname)(this_term::SymbolicTerm, $(args...)) 
            if this_term.operation == :∂
                ((length(this_term.subterms) == 1) && (this_term.subterms[1] isa SymbolicWord)) || error("AST error")
                return true, $eval_local_funcname(this_term.subterms[1], get_VarAttribute(this_term.subterms[1]), $(args...))
            else
                term_changed, subterms = false, this_term.subterms
                for i = 1:length(subterms)
                    subterm_changed, subterms[i] = $eval_global_funcname(subterms[i], $(args...)) 
                    term_changed |= subterm_changed
                end
                return term_changed, refresh_Term(this_term, term_changed)
            end
        end
    end
end

function diff_Eval_Time_Local!(this_word::SymbolicWord, src_attributes::Vector{Symbol}) # time derivative, external var cant have
    if (:EXTERNAL_VAR in src_attributes)
        return 0
    else
        this_word.td_order += 1
        return this_word
    end
end

function diff_Eval_Space_Local!(this_word::SymbolicWord, src_attributes::Vector{Symbol}, d_id::Symbol) # spatial derivative, external var only on the controlpoint can have
    if (:EXTERNAL_VAR in src_attributes) && (~(:CONTROLPOINT_VAR in src_attributes))
        return 0.
    else
        this_word.sd_ids = sort!([this_word.sd_ids; d_id])
        return this_word
    end
end

function diff_Eval_Variation_Local!(this_word::SymbolicWord, src_attributes::Vector{Symbol}, tb::TensorTable) # variational derivative
    if (:INTERNAL_VAR in src_attributes) 
        return construct_Term(:δ, [this_word])
    elseif (:EXTERNAL_VAR in src_attributes)
        return 0.
    else
        return diff_Variation(evaluate_Tensor(tb, this_word), tb)
    end
end

is_variation(x) = false
is_variation(x::SymbolicTerm) = x.operation == :δ
collect_Variations(x::GroundTerm, tb) = collect_Variations(Dict{SymbolicWord, Vector{GroundTerm}}(), x, tb)
collect_Variations(buffer::Dict{SymbolicWord, Vector{GroundTerm}}, x::GroundTerm, tb) = _collect_Variations(buffer, diff_Variation(x, tb))
_collect_Variations(buffer::Dict{SymbolicWord, Vector{GroundTerm}}, this_term::Union{Number, SymbolicWord}) = buffer
function _collect_Variations(buffer::Dict{SymbolicWord, Vector{GroundTerm}}, this_term::SymbolicTerm) 
    term_op = this_term.operation
    if term_op == :δ
        push!(get!(buffer, this_term.subterms[1], GroundTerm[]), 1.)        
    elseif term_op == :+
        for subterm in this_term.subterms
            _collect_Variations(buffer, subterm)
        end
    elseif term_op == :*
        is_var = is_variation.(this_term.subterms)
        var_IDs = findall(is_var)
        length(var_IDs) != 1 && error("One LHS should contain and only contain one variation but not in $(this_term), it is an internal error that shouldn't appear")
        push!(get!(buffer, this_term.subterms[var_IDs[1]].subterms[1], GroundTerm[]), ⨂(this_term.subterms[.~is_var]))
    else
        error("Wrong IR, internal bug")
    end
    return buffer
end

_delta_Func(c1::Number, c2::Number) = c1 == c2 ? 1. : 0.
_delta_Func(c1, c2) = construct_Word(:δ, [c1, c2], IndexSym[])
function diff_Eval_Symbol_Local!(src_word::SymbolicWord, src_attributes::Vector{Symbol}, tb::TensorTable, diff_word::SymbolicWord) # symbolic derivative
    if (src_word.base_variable == diff_word.base_variable) && (src_word.td_order == diff_word.td_order) && (length(src_word.c_ids) == length(diff_word.c_ids)) && (length(src_word.sd_ids) == length(diff_word.sd_ids))
        return ⨂([[_delta_Func(i1, i2) for (i1, i2) in zip(src_word.c_ids, diff_word.c_ids)]; [_delta_Func(i1, i2) for (i1, i2) in zip(src_word.sd_ids, diff_word.sd_ids)]])
    else
        if (:INTERNAL_VAR in src_attributes) || (:EXTERNAL_VAR in src_attributes)
            return 0.
        else
            this_tensorinfo = get_TensorDiff!(tb, word_To_TensorInfo(src_word), word_To_TensorInfo(diff_word))
            return tensorinfo_To_Word(this_tensorinfo, IndexSym[collect_ids(src_word); collect_ids(diff_word)])
        end
    end
end