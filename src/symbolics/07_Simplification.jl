using Base: Number
# for simplification
@Define_Semantic_Constraint F₁ ∊ (isempty(get_FreeIndex(F₁)))

@Define_Rewrite_Rule Add_Splat ≔ ((a...) + (+(b...)) + (c...) => a + b + c)
@Define_Rewrite_Rule Mul_Splat ≔ ((a...) * (*(b...)) * (c...) => a * b * c)
@Define_Rewrite_Rule Pow_Splat ≔ ((a ^ b) ^ c => a ^ (b * c))

@Define_Rewrite_Rule Distributive_MP ≔ ((F₁ * (b...)) ^ c => (a ^ c) * (*(b)) ^ c)
@Define_Rewrite_Rule Distributive_AM ≔ ((a...) * (b + (c...)) * (d...) => a * b * d + a * (+(c)) * d)

expand_And_Flatten(source_term::Union{Number, SymbolicWord}) = source_term
function expand_And_Flatten(source_term::SymbolicTerm) # regulate the term, before merging 
    is_changed, this_term = true, source_term
    while is_changed
        is_changed, this_term = apply_Rules(this_term, [Distributive_MP, Distributive_AM])
        is_changed_1, this_term = apply_Rules(this_term, [Add_Splat, Mul_Splat, Pow_Splat])
        is_changed |= is_changed_1
    end
    return this_term
end

check_Merge(src::Union{Number, SymbolicWord}) = src
function check_Merge(source_term::SymbolicTerm) # term merging is hard-coded via dictionary/hashing
    new_subterms = check_Merge.(source_term.subterms)
    if source_term.operation == :+
        classifier = Dict{GroundTerm, FEM_Float}()
        for subterm in new_subterms
            if subterm isa Number
                main_term = 1.
                factor = subterm
            elseif (subterm isa SymbolicTerm) && (subterm.operation == :*) && (subterm.subterms[1] isa Number)
                main_term = ⨂(subterm.subterms[2:end])
                factor = subterm.subterms[1]
            else
                main_term = subterm
                factor = 1.
            end
            classifier[main_term] = get(classifier, main_term, 0.) + factor
        end
        res_subterms = GroundTerm[]
        for (main_term, factor) in classifier
            if main_term isa SymbolicTerm && main_term.operation == :*
                push!(res_subterms, ⨂([factor; main_term.subterms]))
            else
                push!(res_subterms, ⨂([factor; main_term]))
            end
        end
        return ⨁(res_subterms) # :+ only affect constant factor, so still regular
    elseif source_term.operation == :*
        classifier = Dict{GroundTerm, GroundTerm}()
        
        has_free_idx = map(x -> isempty(get_FreeIndex(x)), new_subterms)
        preserved_terms = new_subterms[.~(has_free_idx)]
        processing_terms = new_subterms[has_free_idx]
        
        for subterm in processing_terms
            if subterm isa Number
                main_term = subterm
                factor = 1.
            elseif (subterm isa SymbolicTerm) && (subterm.operation == :^) 
                main_term, factor = subterm.subterms
            else
                main_term = subterm
                factor = 1.
            end
            classifier[main_term] = ⨁([get(classifier, main_term, 0.), factor])
        end
        processed_terms = [construct_Term(:^, [main_term; check_Merge(factor)]) for (main_term, factor) in classifier] 
        #:* may merge term on the power, e.g. a^b * a^(-b) -> a^1 and may trigger, upper level merging, but itself is still regular
        return ⨂([processed_terms; preserved_terms])
    else
        return construct_Term(source_term.operation, new_subterms)
    end
end
simplify_Common(source_term) = check_Merge(expand_And_Flatten(source_term))

replace_SpecialTerm!(x) = _replace_SpecialTerm!(x) |> simplify_Common
_replace_SpecialTerm!(this_term::Number) = this_term
function _replace_SpecialTerm!(this_term::SymbolicWord) #this function may accomodate more special operators
    @Takeout (base_variable, td_order, c_ids, sd_ids) FROM this_term
    if base_variable == :δ
        if ~((td_order == 0) && isempty(sd_ids))
            return FEM_Float(0.)
        elseif (length(c_ids) == 2) && (c_ids[1] isa Number) && (c_ids[2] isa Number)
            return (c_ids[1] == c_ids[2]) ? FEM_Float(1.) : FEM_Float(0.)
        end
    elseif base_variable == :ϵ
        if length(c_ids) == 3
            if ~((td_order == 0) && isempty(sd_ids))
                return FEM_Float(0.)
            elseif (c_ids[1] isa Number) && (c_ids[2] isa Number) && (c_ids[3] isa Number)
                if (c_ids == [1, 2, 3]) || (c_ids == [2, 3, 1]) || (c_ids == [3, 1, 2])
                    return 1.
                elseif (c_ids == [1, 3, 2]) || (c_ids == [3, 2, 1]) || (c_ids == [2, 1, 3])
                    return -1.
                else
                    return 0.
                end
            end
        end
    end
    return this_term
end
function _replace_SpecialTerm!(this_term::SymbolicTerm)
    for i = 1:length(this_term.subterms)
        this_term.subterms[i] = _replace_SpecialTerm!(this_term.subterms[i])
    end
    return this_term
end
