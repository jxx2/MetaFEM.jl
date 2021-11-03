collect_Index(source) = isempty(source) ? IndexSym[] : collect(source)
get_FreeIndex(this_num::Union{Number, Symbol}) = IndexSym[]
get_FreeIndex(this_word::SymbolicWord) = collect_Index(keys(parse_Word_Index(this_word)[1]))
get_FreeIndex(this_term::SymbolicTerm) = collect_Index(this_term.free_index)

get_DumbIndex(this_num::Union{Number, Symbol}) = IndexSym[]
get_DumbIndex(this_word::SymbolicWord) = collect_Index(keys(parse_Word_Index(this_word)[2]))
get_DumbIndex(this_term::SymbolicTerm) = collect_Index(this_term.dumb_index)

construct_Term(this_num::Number) = FEM_Float(this_num)
function construct_Term(arg::Symbol)
    if arg in keys(VARIABLE_ATTRIBUTES)
        return SymbolicWord(arg, FEM_Int(0), (), ())
    else
        # this_term = Core.eval(@__MODULE__, arg) 
        # this_term = Core.eval(Main, arg) 
        this_term = Base.MainInclude.eval(arg) 
        if this_term isa GroundTerm
            isempty(get_FreeIndex(this_term)) || error("Scalars don't have index")
            return this_term
        elseif this_term isa Number
            return FEM_Float(this_term)
        else
            error(this_term, " is not a scalar")
        end
    end
end

function collect_IndexSym!(arr, source)  
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
            collect_IndexSym!(d_ids, arg.args[2].args[:])
            collect_IndexSym!(c_ids, arg.args[3:end])
        else
            collect_IndexSym!(c_ids, arg.args[2:end])
        end

        if sym in keys(VARIABLE_ATTRIBUTES)
            return construct_Word(sym, c_ids, d_ids)
        else
            # this_term = Core.eval(@__MODULE__, sym) 
            # this_term = Core.eval(Main, sym) 
            this_term = Base.MainInclude.eval(sym) 
            source_free_index = get_FreeIndex(this_term)
            length(source_free_index) == length(c_ids) || error("Wrong index number")

            this_term = isempty(c_ids) ? this_term : substitute_FreeIndex(this_term, source_free_index, c_ids)
            return do_SpatialDiff(this_term, d_ids)
        end
    elseif arg.head == :call
        operation, arguments = arg.args[1], arg.args[2:end]
        return construct_Term(operation, construct_Term.(arguments))
    else
        error("Wrong grammar:")
    end
end
#Note current no shift
function construct_Term(operation::Symbol, source_subterms::Vector)
    # shortcuts
    if operation in [:+, :-]
        subterms = source_subterms[source_subterms .!= 0.]
        isempty(subterms) && return FEM_Float(0.)
        (length(subterms) == 1.) && return (operation == :+ ? subterms[1] : construct_Term(:*, [FEM_Float(-1), subterms[1]]))

        free_index = get_FreeIndex(subterms[1])
        for this_subterm in subterms[2:end]
            new_free_index = get_FreeIndex(this_subterm)
            if Set(free_index) != Set(new_free_index)
                error(visualize(this_subterm), visualize(subterms[1]), " Should have same free index but not")
            end
        end
        return SymbolicTerm(operation, Tuple(subterms), Tuple(free_index), ())
    elseif operation == :*
        subterms = source_subterms[source_subterms .!= 1]
        isempty(subterms) && return FEM_Float(1.)
        (length(subterms) == 1) && return subterms[1]
        (sum(subterms .== 0.) > 0.) && return FEM_Float(0.)
    elseif operation == :^
        subterms = source_subterms
        (subterms[2] == 0.) && return FEM_Float(1.)
        (subterms[2] == 1.) && return subterms[1]
        (subterms[1] == 0.) && return FEM_Float(0.)
        (subterms[1] == 1.) && return FEM_Float(1.)
    elseif operation == :Bilinear
        subterms = source_subterms
        (sum(subterms .== 0.) > 0.) && return FEM_Float(0.)
    else
        subterms = source_subterms
        isempty(subterms) && return SymbolicTerm(operation, (), (), ())
    end

    free_index = get_FreeIndex(subterms[1])
    dumb_index = Symbol[]
    for this_subterm in subterms[2:end]
        new_free_index = get_FreeIndex(this_subterm)
        append!(dumb_index, intersect(free_index, new_free_index))
        symdiff!(free_index, new_free_index)
    end

    total_indices = Set(union(free_index, dumb_index))
    for i = 1:length(subterms)
        src_dumb_indices = get_DumbIndex(subterms[i])
        overlapped_indices = collect(intersect(total_indices, src_dumb_indices))
        if ~isempty(overlapped_indices)
            subterms[i] = substitute_Term(subterms[i], overlapped_indices, [gensym() for i = 1:length(overlapped_indices)])
        end
    end
    return SymbolicTerm(operation, Tuple(subterms), Tuple(free_index), Tuple(dumb_index))
end

substitute_FreeIndex(this_number::Number, matchers::Vector, targets::Vector) = this_number
function substitute_FreeIndex(this_term::GroundTerm, matchers::Vector, targets::Vector)
    src_dumb_indices = get_DumbIndex(this_term)
    overlapped_indices = collect(intersect(src_dumb_indices, targets))
    hygiene_term = isempty(overlapped_indices) ? this_term : substitute_Term(this_term, overlapped_indices, [gensym() for i = 1:length(overlapped_indices)])
    return this_term isa SymbolicTerm ? construct_Term(this_term.operation, collect(substitute_FreeIndex.(this_term.subterms, Ref(matchers), Ref(targets)))) :
                                        substitute_Term(hygiene_term, matchers, targets)
end

mark_Mapping(source, matchers::Vector) = findfirst(x -> x == source, matchers)
substitute_Term(this_number::Number, matchers::Vector, targets::Vector) = this_number
function substitute_Term(this_word::SymbolicWord, matchers::Vector, targets::Vector)
    @Takeout (base_variable, td_order, c_ids, sd_ids) FROM this_word

    match_c_ids, match_sd_ids = collect(c_ids), collect(sd_ids)

    c_markers = mark_Mapping.(match_c_ids, Ref(matchers))
    sd_markers = mark_Mapping.(match_sd_ids, Ref(matchers))

    target_c_ids = map(i -> isnothing(c_markers[i]) ? match_c_ids[i] : targets[c_markers[i]], 1:length(match_c_ids))
    target_sd_ids = map(i -> isnothing(sd_markers[i]) ? match_sd_ids[i] : targets[sd_markers[i]], 1:length(match_sd_ids))

    final_c_ids = Tuple(collect_IndexSym!(IndexSym[], target_c_ids)) 
    final_sd_ids = Tuple(collect_IndexSym!(IndexSym[], target_sd_ids)) 

    return SymbolicWord(base_variable, td_order, final_c_ids, final_sd_ids)
end
substitute_Term(this_term::SymbolicTerm, matchers::Vector, targets::Vector) = construct_Term(this_term.operation, collect(substitute_Term.(this_term.subterms, Ref(matchers), Ref(targets))))

⨁(subterms::Union{Vector, Tuple}) = construct_Term(:+, subterms)
⨂(subterms::Union{Vector, Tuple}) = construct_Term(:*, subterms)

#Note this always returns a single term
unroll_Dumb_Indices(this_number::Number, dim::Integer) = this_number
function unroll_Dumb_Indices(this_word::SymbolicWord, dim::Integer)
    _, dumb_index = parse_Word_Index(this_word)
    isempty(dumb_index) && return this_word

    unrolled_syms = dumb_index |> keys |> collect
    this_iterator = Iterators.product([1:dim for i in unrolled_syms]...)
    matchers = [collect(unrolled_syms) for substituted_indices in this_iterator]
    targets = [collect(substituted_indices) for substituted_indices in this_iterator]
    final_terms = substitute_Term.(Ref(this_word), vec(matchers), vec(targets))
    return ⨁(final_terms)
end

function unroll_Dumb_Indices(this_term::SymbolicTerm, dim::Integer)
    new_term = construct_Term(this_term.operation, collect(unroll_Dumb_Indices.(this_term.subterms, dim)))
    @Takeout operation, dumb_index, subterms FROM new_term

    if ~isempty(dumb_index)
        this_iterator = Iterators.product([1:dim for i in dumb_index]...)
        matcher_groups = [collect(dumb_index) for substituted_indices in this_iterator]
        target_groups = [collect(substituted_indices) for substituted_indices in this_iterator]
        new_term = substitute_Term.(Ref(new_term), matcher_groups, target_groups) |> vec |> ⨁
    end
    return new_term
end
#parsing
parse_Term2Expr(dim::Integer, this_num::FEM_Float) = this_num
parse_Term2Expr(dim::Integer, this_word::SymbolicWord) = word_To_TotalSym(dim, this_word)

const SPECIAL_DOT_OP = Dict(:+ => :.+, :* => :.*, :^ => :.^)
function parse_Term2Expr(dim::Integer, this_term::SymbolicTerm)
    new_subterms = parse_Term2Expr.(dim, this_term.subterms)
    if this_term.operation in keys(SPECIAL_DOT_OP)
        dotted_op = SPECIAL_DOT_OP[this_term.operation]
        ex = new_subterms[1]
        for next_subterm in new_subterms[2:end]
            ex = Expr(:call, dotted_op, ex, next_subterm)
        end
        return ex
    else
        return Expr(:., this_term.operation, Expr(:tuple, new_subterms...))
    end
end
#parsing
do_SpatialDiff(this_term::Number, d_id::IndexSym) = FEM_Float(0)
do_SpatialDiff(this_term::SymbolicWord, d_id::IndexSym) = SymbolicWord(this_term.base_variable, this_term.td_order, this_term.c_ids, (this_term.sd_ids..., d_id))
function do_SpatialDiff(this_term::SymbolicTerm, d_id::IndexSym)
    diff_term = construct_Term(:∂, [this_term; variable])
    while true
        changed_op, diff_term = diff_Operator(diff_term)
        _, diff_term = diff_Spatial(diff_term)
        diff_term = simplify_Basic(diff_term)
        ~(changed_op) && break
    end
    return simplify_Basic(eval_Index_Diff(diff_term))
end
function do_SpatialDiff(this_term, d_ids::Vector)
    for d_id in d_ids
        this_term = do_SpatialDiff(this_term, d_id)
    end
    return this_term
end

eval_Index_Diff(this_term::Union{Number, SymbolicWord}) = this_term
function eval_Index_Diff(this_term::SymbolicTerm)
    if this_term.operation == :∂ && length(this_term.subterms) == 2
        return do_SpatialDiff(this_term.subterms[1], this_term.subterms[2])
    else
        return construct_Term(this_term.operation, eval_Index_Diff.(this_term.subterms))
    end
end
