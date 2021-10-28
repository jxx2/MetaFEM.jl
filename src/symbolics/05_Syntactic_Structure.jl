parse_GeneralTerm(arg::Number) = FEM_Float(arg)
parse_GeneralTerm(arg::GroundTerm) = arg
parse_GeneralTerm(arg::Symbol) = FixSubtermVariable(arg)
function parse_GeneralTerm(arg::Expr)
    if arg.head == :call
        if arg.args[1] isa Symbol
            operation = arg.args[1]
            operation_is_fixed = true
        elseif arg.args[1] isa Expr && arg.args[1].head == :braces
            operation = arg.args[1].args[1]
            operation_is_fixed = false
        end
        subterms = Tuple(parse_GeneralTerm.(arg.args[2:end]))
        return FunctionVariable(operation, operation_is_fixed, subterms)
    elseif arg.head == :...
        return FreeSubtermVariable(arg.args[1])
    elseif arg.head == :vect
        return parse_GeneralTerm(construct_Term(arg.args[1]))
    else
        error("Wrong syntax")
    end
end

get_VarSym(x::GroundTerm) = :_
get_VarSym(x::SubtermVariable) = x.sym
get_VarSym(x::FunctionVariable) = x.operation_is_fixed ? :_ : x.operation

function assemble_Syms(match_nodes::Vector{GeneralTerm})
    node_syms = get_VarSym.(match_nodes)
    syms = setdiff(node_syms, [:_])
    nodes_2_syms = map(x -> x in syms ? findfirst(y -> y == x, syms) : 0, node_syms)
    syms_2_nodes = [findall(x -> x == i, nodes_2_syms) for i = 1:length(syms)]

    is_independent = BitArray(i in getindex.(syms_2_nodes, 1) for i = 1:length(match_nodes))
    return syms, nodes_2_syms, is_independent
end

function assemble_SubNodes(this_node::FunctionVariable, this_node_ID::Integer, matcher_info::Tuple)
    match_nodes, parent_node_IDs = matcher_info

    last_subnode_ID = length(parent_node_IDs)
    subterms = collect(this_node.subterms)
    is_op_nodes = typeof.(subterms) .== FunctionVariable
    op_nodes = subterms[is_op_nodes]
    op_node_IDs = findall(is_op_nodes) .+ last_subnode_ID

    append!(match_nodes, subterms)
    append!(parent_node_IDs, fill(this_node_ID, size(subterms)))
    return op_nodes, op_node_IDs
end

function decide_Undecidable_Freeterm(parent_node_IDs::Vector, match_nodes::Vector{GeneralTerm}, nodes_2_syms::Vector, is_independent::BitArray)
    size_inferences = [(false, 0, 0, FEM_Int[]) for i = 1:length(parent_node_IDs)]
    total_free_node_IDs = findall(x -> x isa FreeSubtermVariable, match_nodes)

    for this_node_ID in unique(parent_node_IDs)
        group_node_IDs = findall(x -> x == this_node_ID, parent_node_IDs)

        group_fix_IDs = setdiff(group_node_IDs, total_free_node_IDs)
        group_freeterm_IDs = intersect(group_node_IDs, total_free_node_IDs)
        group_independent_freeterm_IDs = group_freeterm_IDs[is_independent[group_freeterm_IDs]]

        if ~isempty(group_independent_freeterm_IDs)
            infered_ID = group_independent_freeterm_IDs[end]
            fixnode_number = sum(infered_ID .< group_fix_IDs)

            gp_dep_free_IDs = group_freeterm_IDs[.~is_independent[group_freeterm_IDs]]
            is_self_copy = nodes_2_syms[infered_ID] .== nodes_2_syms[gp_dep_free_IDs]
            self_copy_number = sum(is_self_copy) + 1

            different_dep_free_IDs = gp_dep_free_IDs[.~is_self_copy]
            dependent_freenode_IDs = different_dep_free_IDs[infered_ID .< different_dep_free_IDs]

            size_inferences[infered_ID] = (true, fixnode_number, self_copy_number, dependent_freenode_IDs)
        end
    end
    return size_inferences
end

function assemble_Matcher(this_node::GeneralTerm)
    match_nodes = GeneralTerm[this_node]
    parent_node_IDs = [0]
    matcher_info = (match_nodes, parent_node_IDs)

    op_nodes, op_node_IDs = FunctionVariable[], FEM_Int[]
    if this_node isa FunctionVariable
        push!(op_nodes, this_node)
        push!(op_node_IDs, 1)
    end

    while ~isempty(op_node_IDs)
        next_level_op_nodes, next_level_op_node_IDs = FunctionVariable[], FEM_Int[]
        for (op_node, op_node_ID) in zip(op_nodes, op_node_IDs)
            new_op_nodes, new_op_node_IDs = assemble_SubNodes(op_node, op_node_ID, matcher_info)
            append!(next_level_op_nodes, new_op_nodes)
            append!(next_level_op_node_IDs, new_op_node_IDs)
        end
        op_nodes, op_node_IDs = next_level_op_nodes, next_level_op_node_IDs
    end

    match_nodes, parent_node_IDs = matcher_info
    syms, nodes_2_syms, is_independent = assemble_Syms(match_nodes)

    total_free_node_IDs =  findall(x -> x isa FreeSubtermVariable, match_nodes)

    size_inferences = decide_Undecidable_Freeterm(parent_node_IDs, match_nodes, nodes_2_syms, is_independent)
    return Matcher(match_nodes, parent_node_IDs, syms, nodes_2_syms, is_independent, size_inferences)
end

function define_Rewriting_Structure(this_definition::Expr)
    if this_definition.head == :call && this_definition.args[1] == :(=>)
        structure_to_match = parse_GeneralTerm(this_definition.args[2])
        structure_to_produce = parse_GeneralTerm(this_definition.args[3])

        matcher = assemble_Matcher(structure_to_match)
        return RewritingRule(structure_to_match, structure_to_produce, matcher)
    end
end

macro Define_Rewrite_Rule(input_ex::Expr)
    @assert input_ex.head == :≔
    lhs, rhs = input_ex.args[1], input_ex.args[2]
    rw_rule = define_Rewriting_Structure(rhs)
    return esc(:($lhs = $rw_rule))
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
        define_code = deepcopy(input_ex)
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
