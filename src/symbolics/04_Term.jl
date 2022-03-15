get_FreeIndex(this_num::Union{Number, Symbol}) = Symbol[]
get_FreeIndex(this_word::SymbolicWord) = parse_Word_Index(this_word)[1]
get_FreeIndex(this_term::SymbolicTerm) = copy(this_term.free_index)

get_DumbIndex(this_num::Union{Number, Symbol}) = Symbol[]
get_DumbIndex(this_word::SymbolicWord) = parse_Word_Index(this_word)[2]
get_DumbIndex(this_term::SymbolicTerm) = copy(this_term.dumb_index)

construct_Term(this_num::Number) = FEM_Float(this_num)
construct_Term(arg::Symbol) = SymbolicWord(arg)

function convert_And_Append!(arr, source)
    for id in source
        if id isa Number
            push!(arr, FEM_Int(id))
        elseif id isa Symbol
            push!(arr, id)
        else
            error("Wrong index")
        end
    end
    return arr
end

function construct_Term(arg::Expr)
    if arg.head == :curly
        sym = arg.args[1]
        length(arg.args) == 1 && error("Wrong grammar: {indices} cannot be empty, not beautiful")
        c_ids, d_ids = IndexSym[], IndexSym[]
        if typeof(arg.args[2]) == Expr && arg.args[2].head == :parameters
            convert_And_Append!(d_ids, arg.args[2].args[:])
            convert_And_Append!(c_ids, arg.args[3:end])
        else
            convert_And_Append!(c_ids, arg.args[2:end])
        end
        return construct_Word(sym, c_ids, d_ids)
    elseif arg.head == :call
        operation, arguments = arg.args[1], arg.args[2:end]
        return construct_Term(operation, construct_Term.(arguments))
    else
        error("Unknown grammar")
    end
end
#Note current no shift
function construct_Term(operation::Symbol, subterms::Vector)
    # shortcuts
    if operation == :+
        nums = filter(x -> x isa Number, subterms)
        num = isempty(nums) ? 0. : sum(nums)
        filter!(x -> ~(x isa Number), subterms)
        isempty(subterms) && return num
        if num != 0 
            subterms = [num; subterms]
        else
            (length(subterms) == 1) && return subterms[1]
        end
        subterms = subterms[sortperm(hash.(subterms))]  #:+ dont need to be sorted since does not matter, no, it need to be sorted
        
        free_index = get_FreeIndex(subterms[1])
        for this_subterm in subterms[2:end]
            new_free_index = get_FreeIndex(this_subterm)
            (sort(free_index) != sort(new_free_index)) && error("$(visualize(this_subterm)), $(visualize(subterms[1])) should have the same free index but not, $new_free_index, $free_index")
        end
        return SymbolicTerm(operation, convert(Vector{Union{Number, Symbol, SymbolicWord, SymbolicTerm}}, subterms), free_index, Symbol[])
    elseif operation == :*
        nums = filter(x -> x isa Number, subterms)
        num = isempty(nums) ? 1. : prod(nums)
        filter!(x -> ~(x isa Number), subterms)
        isempty(subterms) && return num
        if num == 0 
            return num
        elseif num == 1
            (length(subterms) == 1) && return subterms[1]
        else
            subterms = [num; subterms] 
        end
        subterms = subterms[sortperm(hash.(subterms))] # :* need to be sourted for merging in plus
    elseif operation == :^
        (subterms[1] isa Number) && (subterms[2] isa Number) && return FEM_Float(subterms[1] ^ subterms[2])
        (subterms[2] == 0.) && return FEM_Float(1.)
        (subterms[2] == 1.) && return subterms[1]
        (subterms[1] == 0.) && return FEM_Float(0.)
        (subterms[1] == 1.) && return FEM_Float(1.)
        if isempty(get_FreeIndex(subterms[1])) && isempty(get_FreeIndex(subterms[2]))
            return SymbolicTerm(:^, subterms, Symbol[], Symbol[])
        else
            error("Free index in power: $subterms, is ambiguous")
        end
    elseif operation == :Bilinear
        (length(subterms) == 2) || error("$subterms does not have 2 elements as the subterms of a bilinear term")
        (subterms[1] isa Number) && return FEM_Float(0.)
        (subterms[2] == 0.) && return FEM_Float(0.)
    elseif isempty(subterms)
        return SymbolicTerm(operation, Any[], Symbol[], Symbol[])
    elseif operation == :-
        if length(subterms) == 1 
            return ⨂([FEM_Float(-1), subterms[1]])
        elseif length(subterms) == 2
            return ⨁([subterms[1], ⨂([FEM_Float(-1), subterms[2]])])
        else
            error("Minus can at most only have 2 subterms")
        end 
    elseif operation == :/
        return ⨂([subterms[1], construct_Term(:^, [subterms[2], FEM_Float(-1)])])
    end

    free_index = get_FreeIndex(subterms[1])
    dumb_index = Symbol[]
    for this_subterm in subterms[2:end]
        for this_index in get_FreeIndex(this_subterm)
            if this_index in dumb_index
                error("$this_index appears 3 times")
            elseif this_index in free_index
                push!(dumb_index, this_index)
                filter!(x -> x != this_index, free_index)
            else
                push!(free_index, this_index)
            end
        end
    end
    total_indices = Symbol[free_index; dumb_index]
    for i in 1:length(subterms)
        for ID in total_indices
            if ID in get_DumbIndex(subterms[i])
                subterms[i] = _substitute_Term!(subterms[i], ID => gensym())
            end
        end
    end
    return SymbolicTerm(operation, subterms, free_index, dumb_index)
end

refresh_Term(this_term, term_changed::Bool) = term_changed ? refresh_Term(this_term) : this_term
refresh_Term(this_term) = this_term
refresh_Term(this_term::SymbolicTerm) = construct_Term(this_term.operation, this_term.subterms)

_substitute_Term!(this_number::Number, mapping::Pair) = this_number #substitute index, may change free/dumb index relations but not term structures
function _substitute_Term!(this_word::SymbolicWord, mapping::Pair)
    source_id, target_id = mapping
    if (source_id in get_FreeIndex(this_word)) && (target_id isa Symbol) && (target_id in get_DumbIndex(this_word))
        this_word = _substitute_Term!(this_word, target_id => gensym())
    end

    if source_id in this_word.c_ids
        replace!(this_word.c_ids, mapping)
        if length(this_word.c_ids) == 2
            (:SYMMETRIC_TENSOR in get_VarAttribute(this_word)) && sort!(this_word.c_ids) #More symmetry, i.e., Voigt ones can be added here
        end
    end # note c_ids and sd_ids are two independent parts

    if source_id in this_word.sd_ids
        sort!(replace!(this_word.sd_ids, mapping))
    end
    this_word
end

function _substitute_Term!(this_term::SymbolicTerm, mapping::Pair)
    source_id, target_id = mapping
    if (source_id in this_term.free_index) || (source_id in this_term.dumb_index)
        if (target_id isa Symbol) && (target_id in this_term.dumb_index)
            this_term = _substitute_Term!(this_term, target_id => gensym())
        end

        for i = 1:length(this_term.subterms)
            this_term.subterms[i] = _substitute_Term!(this_term.subterms[i], mapping)
        end
        return refresh_Term(this_term)
    else
        return this_term
    end
end

# function _substitute_Term!(this_term::SymbolicTerm, mapping::Pair{Symbol, Symbol})
#     src, target = mapping
#     if src in this_term.free_index
#         if target in this_term.dumb_index
#             _substitute_Term!(this_term, target => gensym())
#         elseif target in this_term.free_index
#             setdiff!(this_term.free_index, [src, target])
#             push!(this_term.dumb_index, target)
#         else
#             replace!(this_term.free_index, mapping)
#         end
#     elseif src in this_term.dumb_index
#         replace!(this_term.dumb_index, mapping)
#     else
#         return this_term
#     end

#     for this_subterm in this_term.subterms
#         _substitute_Term!(this_subterm, mapping)
#     end
#     this_term
# end
# function _substitute_Term!(this_term::SymbolicTerm, mapping::Pair{Symbol, FEM_Int})
#     src, _ = mapping
#     if src in this_term.free_index
#         filter!(x -> x != src, this_term.free_index)
#     elseif src in this_term.dumb_index
#         filter!(x -> x != src, this_term.dumb_index)
#     else
#         return this_term
#     end

#     for this_subterm in this_term.subterms
#         _substitute_Term!(this_subterm, mapping)
#     end
#     this_term
# end

function substitute_Term!(this_term, source_ids::Vector, target_ids::Vector)
    mappings = Pair[]
    for (src_id, target_id) in zip(source_ids, target_ids)
        (src_id == target_id) && continue
        if target_id isa Number
            this_term = _substitute_Term!(this_term, src_id => target_id)
        else
            placeholder = gensym()
            this_term = _substitute_Term!(this_term, src_id => placeholder)
            push!(mappings, placeholder => target_id)
        end
    end

    for mapping in mappings
        this_term = _substitute_Term!(this_term, mapping)
    end
    this_term
end

const DEFALUT_INDEX_POOL = [:i, :j, :k, :l, :m, :n, :o, :p, :q, :r, :s, :t]
generate_Index(n::Int) = n < length(DEFALUT_INDEX_POOL) ? DEFALUT_INDEX_POOL[1:n] : [DEFALUT_INDEX_POOL[1:n]; [Symbol("i$j") for j in 1:(length(DEFALUT_INDEX_POOL) - n)]]
function reindex_Term!(this_term, source_ids::Vector) 
    target_ids = generate_Index(length(source_ids))
    return target_ids, substitute_Term!(this_term, source_ids, target_ids)
end

⨁(subterms::Vector) = construct_Term(:+, subterms)
⨂(subterms::Vector) = construct_Term(:*, subterms)
#Note this always returns a new single term
unroll_Dumb_Indices(this_number::Number, dim::Integer) = this_number
function unroll_Dumb_Indices(this_word::SymbolicWord, dim::Integer)
    dumb_index = get_DumbIndex(this_word)
    isempty(dumb_index) && return this_word

    total_iterator = Iterators.product([1:dim for i in dumb_index]...)
    result_subterms = [deepcopy(this_word) for i in total_iterator]
    for ids in total_iterator
        result_subterms[ids...] = substitute_Term!(result_subterms[ids...], dumb_index, [FEM_Int(id) for id in ids])
    end
    return ⨁(vec(result_subterms))
end
function unroll_Dumb_Indices(this_term::SymbolicTerm, dim::Integer)
    @Takeout (subterms, dumb_index) FROM this_term
    this_term.subterms .= unroll_Dumb_Indices.(subterms, dim)
    isempty(dumb_index) && return this_term
    total_iterator = Iterators.product([1:dim for i in dumb_index]...)
    result_subterms = [deepcopy(this_term) for i in total_iterator]
    for ids in total_iterator
        result_subterms[ids...] = substitute_Term!(result_subterms[ids...], dumb_index, [FEM_Int(id) for id in ids])
    end
    return ⨁(vec(result_subterms))
end