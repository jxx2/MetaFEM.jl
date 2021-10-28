function match_GeneralTerm(this_node::GroundTerm, check_terms::Tuple, matcher::Matcher, matching_info::Vector)
    isempty(check_terms) && return false, ()
    return check_terms[1] == this_node, check_terms[2:end]
end

function match_GeneralTerm(this_node::FixSubtermVariable, check_terms::Tuple, matcher::Matcher, matching_info::Vector)
    @Takeout (syms, nodes_2_syms, is_independent) FROM matcher
    node_ID, matched_syms, _, _ = matching_info

    isempty(check_terms) && return false, ()
    this_sym_ID = nodes_2_syms[node_ID]
    this_term = check_terms[1]
    if is_independent[node_ID] #new term node
        checkfunc = get(SEMANTIC_CONSTRAINT, syms[this_sym_ID], always_True)
        is_match = checkfunc(this_term)
        is_match && (matched_syms[this_sym_ID] = this_term)
    else
        is_match = this_term == matched_syms[this_sym_ID]
    end
    return is_match, check_terms[2:end]
end

function match_GeneralTerm(this_node::FunctionVariable, check_terms::Tuple, matcher::Matcher, matching_info::Vector)
    @Takeout (syms, nodes_2_syms, is_independent) FROM matcher
    node_ID, matched_syms, matched_op_nodes, _ = matching_info

    isempty(check_terms) && return false, ()
    this_sym_ID = nodes_2_syms[node_ID]
    this_term = check_terms[1]
    this_term isa SymbolicTerm || return false, ()
    if this_node.operation_is_fixed
        is_match = this_term.operation == this_node.operation
    else
        if is_independent[node_ID] #new func node
            checkfunc = get(SEMANTIC_CONSTRAINT, syms[this_sym_ID], always_True)
            is_match = checkfunc(this_term.operation)
            is_match && (matched_syms[this_sym_ID] = this_term.operation)
        else
            is_match = this_term.operation == matched_syms[this_sym_ID]
        end
    end
    is_match && (matched_op_nodes[node_ID] = this_term)
    return is_match, check_terms[2:end]
end

function match_GeneralTerm(this_node::FreeSubtermVariable, check_terms::Tuple, matcher::Matcher, matching_info::Vector)
    @Takeout (nodes_2_syms, is_independent, size_inferences) FROM matcher
    node_ID, matched_syms, _, undecidable_node_infos = matching_info

    this_sym_ID = nodes_2_syms[node_ID]
    max_subterm_length = length(check_terms)

    if is_independent[node_ID] #new free node
        size_inference = size_inferences[node_ID]

        if size_inference[1] #size can be inferred

            fixnode_number, self_copy_number, dependent_freenode_IDs = size_inference[2:end]
            prematched_dependent_freenodes = matched_syms[nodes_2_syms[dependent_freenode_IDs]]
            preoccupied_number = fixnode_number + sum(length.(prematched_dependent_freenodes))

            this_subterm_length = (max_subterm_length - preoccupied_number) / self_copy_number
            (is_match = isinteger(this_subterm_length)) || return false, ()
        else # size cant be inferred and need to push the undecidable_node_infos
            this_subterm_length = node_ID == undecidable_node_infos[end][1] ? pop!(undecidable_node_infos)[3] + 1 : 0
            is_match = (this_subterm_length <= max_subterm_length) || return false, ()
            push!(undecidable_node_infos, (node_ID, max_subterm_length, this_subterm_length))
        end
        matched_syms[this_sym_ID] = check_terms[1:this_subterm_length]
    else #old free node
        matched_subterms = matched_syms[this_sym_ID]
        this_subterm_length = length(matched_subterms)
        is_match = (this_subterm_length <= max_subterm_length) && check_terms[1:this_subterm_length] == matched_subterms
    end
    return is_match, check_terms[(1 + this_subterm_length):end]
end

function check_Match(source_term::Union{Number, SymbolicWord}, matcher::Matcher)
    @Takeout (match_nodes, syms) FROM matcher
    length(match_nodes) > 1 && return false, Union{GroundTerm, Tuple}[]
    matching_info = [1, Vector{Union{GroundTerm, Symbol, Tuple}}(undef, length(syms)),
                        Vector{Union{GroundTerm, Tuple}}(undef, length(match_nodes)),
                        Tuple{Vararg{FEM_Int, 3}}[(0, 0, 0)]]
    is_match, _ = match_GeneralTerm(match_nodes[1], (source_term,), matcher, matching_info)
    return is_match, matching_info[2]
end

function changing_Failed_Branch(matcher::Matcher, matching_info::Vector)
    _, matched_syms, matched_op_nodes, undecidable_node_infos = matching_info
    node_ID, rest_subterm_length, this_subterm_length = undecidable_node_infos[end]

    node_ID == 0 && return false, ()
    matching_info[1] = node_ID

    matched_syms[matcher.nodes_2_syms[node_ID]:end] .= Ref(())
    matched_op_nodes[node_ID:end] .= Ref(())

    parent_node_ID = matcher.parent_node_IDs[node_ID]
    new_check_terms = matched_op_nodes[parent_node_ID].subterms[(end - rest_subterm_length + 1):end]

    is_match, new_subterms = match_GeneralTerm(matcher.match_nodes[node_ID], new_check_terms, matcher, matching_info)
    return is_match ? (true, new_subterms) : changing_Failed_Branch(matcher, matching_info)
end

function check_Match(source_term::SymbolicTerm, matcher::Matcher)
    @Takeout (match_nodes, syms, parent_node_IDs) FROM matcher
    matching_info = [0, Vector{Union{GroundTerm, Symbol, Tuple}}(undef, length(syms)), #Symbol for operator, tuple for free terms
                        Vector{Union{GroundTerm, Tuple}}(undef, length(match_nodes)),
                        Tuple{Vararg{FEM_Int, 3}}[(0, 0, 0)]]

    check_terms = (source_term,)
    while true
        matching_info[1] += 1
        this_node = match_nodes[matching_info[1]]
        node_match, check_terms = match_GeneralTerm(this_node, check_terms, matcher, matching_info)
        if ~(node_match)
            branch_success, check_terms = changing_Failed_Branch(matcher, matching_info)
            branch_success || return false, matching_info
        end
        if isempty(check_terms)
            if matching_info[1] == length(match_nodes)
                return true, matching_info[2]
            elseif parent_node_IDs[matching_info[1]] != parent_node_IDs[matching_info[1] + 1]
                parent_node_ID = parent_node_IDs[matching_info[1] + 1]
                check_terms = matching_info[3][parent_node_ID].subterms
            end
        else
            if matching_info[1] == length(match_nodes) || parent_node_IDs[matching_info[1]] != parent_node_IDs[matching_info[1] + 1]
                branch_success, check_terms = changing_Failed_Branch(matcher, matching_info)
                branch_success || return false, matching_info
            end
        end
    end
end

embody_GeneralTerm(this_node::GroundTerm, this_match::Dict) = this_node
function embody_GeneralTerm(this_node::SubtermVariable, this_match::Dict)
    if this_node.sym in keys(this_match)
        return this_match[this_node.sym]
    elseif this_node.sym in keys(AUX_SYM_DEFINITION)
        this_func, syntactic_symbols = AUX_SYM_DEFINITION[this_node.sym]
        semantic_symbols = [this_match[sym] for sym in syntactic_symbols]
        return this_func(semantic_symbols...)
    else
        error("Wrong syntax")
    end
end

function embody_GeneralTerm(this_node::FunctionVariable, this_match::Dict)
    semantic_subterms = GroundTerm[]
    for this_syntactic_subnode in this_node.subterms
        this_semantic_subterm = embody_GeneralTerm(this_syntactic_subnode, this_match)
        if this_semantic_subterm isa GroundTerm
            push!(semantic_subterms, this_semantic_subterm)
        else
            append!(semantic_subterms, this_semantic_subterm)
        end
    end
    syntax_operation = this_node.operation
    if this_node.operation_is_fixed
        return construct_Term(syntax_operation, semantic_subterms)
    else
        if syntax_operation in keys(this_match)
            return construct_Term(this_match[syntax_operation], semantic_subterms)
        elseif syntax_operation in keys(AUX_SYM_DEFINITION)
            this_func, syntactic_symbols = AUX_SYM_DEFINITION[syntax_operation]
            semantic_symbols = [this_match[sym] for sym in syntactic_symbols]
            return construct_Term(this_func(semantic_symbols...), semantic_subterms)
        end
    end
end

function apply_Rule_OneNode(source_term::GroundTerm, this_rule::RewritingRule)
    @Takeout (matcher, structure_to_produce) FROM this_rule
    is_match, matched_syms = check_Match(source_term, matcher)
    return is_match ? (true, embody_GeneralTerm(structure_to_produce, Dict(matcher.syms .=> matched_syms))) : (false, source_term)
end

apply_Rules_OneNode(source_term::Integer, rules::Vector{RewritingRule}) = error(source_term, visualize.(rules))
function apply_Rules_OneNode(source_term::GroundTerm, rules::Vector{RewritingRule})
     term_changed = false
     this_term = source_term
     for this_rule in rules
         #Note the judgement term_changed is preserved for collecting history
         this_check, this_term = apply_Rule_OneNode(this_term, this_rule)
         term_changed |= this_check
     end
     return term_changed, this_term
end

apply_Rules_All_Node(source_term::Union{FEM_Float, SymbolicWord}, rules::Vector{RewritingRule}) = apply_Rules_OneNode(source_term, rules)
function apply_Rules_All_Node(source_term::SymbolicTerm, rules::Vector{RewritingRule})
    recursive_result = apply_Rules_All_Node.(source_term.subterms, Ref(rules))
    middle_check = ~prod(.~getindex.(recursive_result, 1))
    new_subterms = getindex.(recursive_result, 2)

    intermediate_term = middle_check ? construct_Term(source_term.operation, collect(new_subterms)) : source_term
    final_check, final_term = apply_Rules_OneNode(intermediate_term, rules)
    return final_check || middle_check, final_term
end

apply_Rules_Recursive(source_term::Union{FEM_Float, SymbolicWord}, rules::Vector{RewritingRule}) = apply_Rules_OneNode(source_term, rules)
function apply_Rules_Recursive(source_term::SymbolicTerm, rules::Vector{RewritingRule})
    term_changed, this_term = apply_Rules_All_Node(source_term, rules)
    return term_changed ? (true, apply_Rules_Recursive(this_term, rules)[2]) : (false, source_term)
end

macro Compile_Rewrite_Rules(func_name, rule_names)
    rule_groups = vectorize_Args(rule_names)
    merged_rules = Expr(:vcat)
    for rule_group in rule_groups
        push!(merged_rules.args, :(($rule_group)...))
    end
    return esc(:(
    $func_name(source_term::GroundTerm) = apply_Rules_Recursive(source_term, $merged_rules)
    ))
end
