parse_GeneralTerm(arg::Number) = FEM_Float(arg)
parse_GeneralTerm(arg::GroundTerm) = arg
parse_GeneralTerm(arg::Symbol) = SubtermVariable(arg, IDPDT_SINGLE())
function parse_GeneralTerm(arg::Expr)
    if arg.head == :call
        if arg.args[1] isa Symbol
            operation = arg.args[1]
            tag = FIXED_OP()
        elseif (arg.args[1] isa Expr) && (arg.args[1].head == :braces)
            operation = arg.args[1].args[1]
            tag = IDPDT_OP()
        else
            error("Wrong rewriting rule syntax")
        end
        subterms = parse_GeneralTerm.(arg.args[2:end])
        return FunctionVariable(operation, tag, subterms)
    elseif arg.head == :...
        return SubtermVariable(arg.args[1], IDPDT_FREE())
    elseif arg.head == :vect
        return parse_GeneralTerm(construct_Term(arg.args[1])) # construct_Term returns a ground term
    else
        error("Wrong rewriting rule syntax")
    end
end

get_Sym(this_node::FunctionVariable) = this_node.operation
get_Sym(this_node::SubtermVariable) = this_node.sym

get_Tag(this_node::GroundTerm) = FIXED_SINGLE() #place holder, not used since there's no branching need
get_Tag(this_node::Union{FunctionVariable, SubtermVariable}) = this_node.tag

function assemble_Matcher(head_node::GeneralTerm)
    # build matcher structure
    matcher_nodes, node_parent_IDs, node_tail_length, match_batches = GeneralTerm[head_node], Int8[0], Int8[0], UnitRange[]
    build_MatcherStructure!(head_node, 1, matcher_nodes, node_parent_IDs, node_tail_length, match_batches)
    # collect the symbols and retag
    node_sym_IDs = zeros(Int8, length(matcher_nodes))
    syms, sym_constraints = Symbol[], Function[]
    collect_MatcherSyms!(head_node, 1, node_sym_IDs, syms, sym_constraints)

    for subterm_IDs in match_batches
        for this_subterm_ID in subterm_IDs
            collect_MatcherSyms!(matcher_nodes[this_subterm_ID], this_subterm_ID, node_sym_IDs, syms, sym_constraints)
        end
        batch_tags = get_Tag.(matcher_nodes[subterm_IDs])
        
        inferrable_independent_node_id = findlast(batch_tags .== IDPDT_FREE())
        if ~isnothing(inferrable_independent_node_id)
            infer_ID = subterm_IDs[inferrable_independent_node_id]
            matcher_nodes[infer_ID].tag = IDPDT_INFER()
        end
    end
    return Matcher(VTuple(GeneralTerm)(matcher_nodes), VTuple(Int8)(node_sym_IDs), VTuple(Int8)(node_parent_IDs), VTuple(Int8)(node_tail_length), VTuple(Symbol)(syms), VTuple(Function)(sym_constraints))
end 

build_MatcherStructure!(args...) = nothing
function build_MatcherStructure!(this_node::FunctionVariable, node_ID::Integer, matcher_nodes::Vector{GeneralTerm}, node_parent_IDs::Vector{Int8}, node_tail_length::Vector{Int8}, match_batches::Vector{UnitRange})
    subterm_length = length(this_node.subterms)
    subterm_range = (1:subterm_length) .+ length(matcher_nodes)
    push!(match_batches, subterm_range)
    append!(matcher_nodes, this_node.subterms)
    append!(node_parent_IDs, [node_ID for i = 1:subterm_length])
    append!(node_tail_length, [subterm_length - i for i = 1:subterm_length])
    for this_subterm_ID in subterm_range #sequential
        build_MatcherStructure!(matcher_nodes[this_subterm_ID], this_subterm_ID, matcher_nodes, node_parent_IDs, node_tail_length, match_batches)
    end
    return nothing
end

collect_MatcherSyms!(args...) = nothing
function collect_MatcherSyms!(this_node::Union{FunctionVariable, SubtermVariable}, node_ID::Integer, node_sym_IDs::Vector{Int8}, syms::Vector{Symbol}, sym_constraints::Vector{Function})
    this_node.tag isa FIXED_OP && return nothing
    is_independent, sym_ID = check_Sym_Independent!(get_Sym(this_node), syms, sym_constraints)
    is_independent || retag_Dependent!(this_node.tag, this_node)
    node_sym_IDs[node_ID] = sym_ID
    nothing
end

function check_Sym_Independent!(this_sym::Symbol, syms::Vector{Symbol}, sym_constraints::Vector{Function})
    ID = findfirst(x -> x == this_sym, syms)
    if isnothing(ID) # new (independent) sym
        push!(syms, this_sym)
        push!(sym_constraints, get(SEMANTIC_CONSTRAINT, this_sym, always_True))
        return true, length(syms)
    else  # old (dependent) sym
        return false, ID
    end
end
retag_Dependent!(args...) = nothing
retag_Dependent!(::IDPDT_OP, this_node) = (this_node.tag = DPDT_OP()); nothing
retag_Dependent!(::IDPDT_SINGLE, this_node) = (this_node.tag = DPDT_SINGLE()); nothing
retag_Dependent!(::IDPDT_FREE, this_node) = (this_node.tag = DPDT_FREE()); nothing

function allocate_MatchingInfo(matcher::Matcher)
    @Takeout (matcher_nodes, syms) FROM matcher
    matched_parent_nodes = Vector{SymbolicTerm}(undef, length(matcher_nodes))
    matched_sym_nodes = Vector{Union{Symbol, GroundTerm, Vector}}(undef, length(syms))
    current_branch = 1
    idp_free_node_num = sum(get_Tag.(matcher_nodes) .== IDPDT_FREE())
    branching_infos = zeros(FEM_Int, 3, idp_free_node_num)
    return @Construct MatchingInfo
end

function define_Rewriting_Structure(this_definition::Expr)
    if this_definition.head == :call && this_definition.args[1] == :(=>)
        structure_to_match = parse_GeneralTerm(this_definition.args[2])
        structure_to_produce = parse_GeneralTerm(this_definition.args[3])
        matcher = assemble_Matcher(structure_to_match)
        matchinginfo = allocate_MatchingInfo(matcher)
        return @Construct RewritingRule
    else
        error("Parse error, $this_definition")
    end
end

macro Define_Rewrite_Rule(input_ex::Expr)
    @assert input_ex.head == :≔
    lhs, rhs = input_ex.args[1], input_ex.args[2]
    return esc(:($lhs = define_Rewriting_Structure($(Meta.quot(rhs)))))
end

always_True(x) = true
SEMANTIC_CONSTRAINT = Dict{Symbol, Function}() #Input symbol to a bool function to check whether the semantic is valid
AUX_SYM_DEFINITION = Dict{Symbol, Tuple{Function, Tuple{Vararg{Symbol}}}}() #Auxiliary symbol to a function and initial symbols to calculate the aux sym semantic

macro Define_Semantic_Constraint(input_ex::Expr)
    if input_ex.head == :call && input_ex.args[1] == :(∊)
        lhs, rhs = input_ex.args[2:3]
        (lhs isa Symbol) || error("Wrong syntax")
        func_name = gensym()

        output_ex = :(begin
            $func_name($lhs) = $rhs
            SEMANTIC_CONSTRAINT[$(Meta.quot(lhs))] = $func_name
        end)
        return esc(output_ex)
    else
        error("Wrong syntax")
    end
end

macro Define_Aux_Semantics(input_ex::Expr)
    if input_ex.head == :(=)
        lhs, _ = input_ex.args
        lhs.head == :call || error("Wrong syntax")
        aux_symbol, syntactic_symbols = lhs.args[1], lhs.args[2:end]

        func_name = gensym()
        define_code = input_ex
        define_code.args[1].args[1] = func_name

        output_ex = :(begin
            $define_code
            AUX_SYM_DEFINITION[$(Meta.quot(aux_symbol))] = ($func_name, tuple($((Meta.quot.(syntactic_symbols))...)))
        end)
        return esc(output_ex)
    else
        error("Wrong syntax")
    end
end