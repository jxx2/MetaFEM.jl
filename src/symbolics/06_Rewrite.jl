function match_Global(source_term, matcher::Matcher, matchinginfo::MatchingInfo)
    matchinginfo.current_branch = 1
    node_ID, last_host_subterm_id = 1, 0
    subterms = [source_term]
    is_success = true
    matcher_size = length(matcher.matcher_nodes)
    while true
        if is_success
            (is_success, matched_size) = match_Local!(matcher.matcher_nodes[node_ID], node_ID, last_host_subterm_id, subterms, matcher, matchinginfo)
        else
            target_branch = (matchinginfo.current_branch -= 1)
            (target_branch == 0) && return false # out of stacks, fail
            node_ID, last_host_subterm_id, _ = matchinginfo.branching_infos[:, target_branch]

            subterms = reset_subterms(node_ID, matcher, matchinginfo)
            (is_success, matched_size) = match_Subterm!(IDPDT_FREE(), node_ID, last_host_subterm_id, subterms, matcher, matchinginfo, false) 
        end

        is_success || continue

        if matcher.node_tail_length[node_ID] == 0 # is node tail, next should switch
            if (matched_size + last_host_subterm_id) == length(subterms)
                (node_ID == matcher_size) && return true #success

                last_host_subterm_id = 0
                subterms = reset_subterms(node_ID += 1, matcher, matchinginfo)
            else
                is_success = false
            end
        else
            node_ID += 1
            last_host_subterm_id += matched_size
        end
    end
end
get_matched_node(node_ID::Integer, matcher::Matcher, matchinginfo::MatchingInfo) = matchinginfo.matched_sym_nodes[matcher.node_sym_IDs[node_ID]]
reset_subterms(node_ID::Integer, matcher::Matcher, matchinginfo::MatchingInfo) = matchinginfo.matched_parent_nodes[matcher.node_parent_IDs[node_ID]].subterms

function match_Local!(this_node::GroundTerm, node_ID::Integer, last_host_subterm_id::Integer, subterms::Vector, matcher::Matcher, matchinginfo::MatchingInfo) 
    (last_host_subterm_id == length(subterms)) && return (false, 1)
    return (this_node == subterms[last_host_subterm_id + 1]), 1
end
function match_Local!(this_node::FunctionVariable, node_ID::Integer, last_host_subterm_id::Integer, subterms::Vector, matcher::Matcher, matchinginfo::MatchingInfo) 
    (last_host_subterm_id == length(subterms)) && return (false, 1)
    target = subterms[last_host_subterm_id + 1]
    (target isa SymbolicTerm) || return (false, 1)
    matchinginfo.matched_parent_nodes[node_ID] = target
    return match_Function!(this_node.tag, this_node.operation, node_ID, target.operation, matcher, matchinginfo), 1
end
match_Local!(this_node::SubtermVariable, node_ID::Integer, last_host_subterm_id::Integer, subterms::Vector, matcher::Matcher, matchinginfo::MatchingInfo) = 
match_Subterm!(this_node.tag, node_ID, last_host_subterm_id, subterms, matcher, matchinginfo)

match_Function!(::FIXED_OP, node_op::Symbol, node_ID::Integer, target_op::Symbol, matcher::Matcher, matchinginfo::MatchingInfo) = node_op == target_op
function match_Function!(::IDPDT_OP, node_op::Symbol, node_ID::Integer, target_op::Symbol, matcher::Matcher, matchinginfo::MatchingInfo) 
    matchinginfo.matched_sym_nodes[matcher.node_sym_IDs[node_ID]] = target_op
    return checker(target_op)
end
match_Function!(::DPDT_OP, node_op::Symbol, node_ID::Integer, target_op::Symbol, checker::Function, matchinginfo::MatchingInfo) = 
get_matched_node(node_ID, matcher, matchinginfo) == target_op

function match_Subterm!(::IDPDT_SINGLE, node_ID::Integer, last_host_subterm_id::Integer, subterms::Vector, matcher::Matcher, matchinginfo::MatchingInfo) 
    (last_host_subterm_id == length(subterms)) && return (false, 1) 
    sym_ID = matcher.node_sym_IDs[node_ID]
    matchinginfo.matched_sym_nodes[sym_ID] = subterms[last_host_subterm_id + 1]
    checker = matcher.sym_constraints[sym_ID]
    return checker(subterms[last_host_subterm_id + 1]), 1
end

function match_Subterm!(::DPDT_SINGLE, node_ID::Integer, last_host_subterm_id::Integer, subterms::Vector, matcher::Matcher, matchinginfo::MatchingInfo)
    (last_host_subterm_id == length(subterms)) && return (false, 1)
    return (get_matched_node(node_ID, matcher, matchinginfo) == subterms[last_host_subterm_id + 1]), 1
end

function match_Subterm!(::DPDT_FREE, node_ID::Integer, last_host_subterm_id::Integer, subterms::Vector, matcher::Matcher, matchinginfo::MatchingInfo) 
    prev_matched_nodes = get_matched_node(node_ID, matcher, matchinginfo)
    prev_length = length(prev_matched_nodes)

    ((last_host_subterm_id + prev_length) <= length(subterms)) || return (false, prev_length)
    return (prev_matched_nodes == subterms[last_host_subterm_id .+ (1:prev_length)]), prev_length
end

function match_Subterm!(::IDPDT_FREE, node_ID::Integer, last_host_subterm_id::Integer, subterms::Vector, matcher::Matcher, matchinginfo::MatchingInfo, is_first_entry::Bool = true) 
    @Takeout (current_branch, branching_infos) FROM matchinginfo

    if is_first_entry
        branching_infos[1, current_branch] = node_ID
        branching_infos[2, current_branch] == last_host_subterm_id
        new_length = (branching_infos[3, current_branch] = 0)
    else
        new_length = (branching_infos[3, current_branch] += 1)
    end
        
    sym_ID = matcher.node_sym_IDs[node_ID]
    occupied_space, infer_num = tail_Space(node_ID, matcher.node_tail_length[node_ID], sym_ID, matcher, matchinginfo) 
    rest_target_length = length(subterms) - last_host_subterm_id - occupied_space

    if (new_length * infer_num) <= rest_target_length
        matchinginfo.matched_sym_nodes[sym_ID] = subterms[last_host_subterm_id .+ (1:new_length)]
        matchinginfo.current_branch += 1
        return true, new_length # no practical need to check free term
    else
        return false, 0
    end
end

function match_Subterm!(::IDPDT_INFER, node_ID::Integer, last_host_subterm_id::Integer, subterms::Vector, matcher::Matcher, matchinginfo::MatchingInfo) 
    sym_ID = matcher.node_sym_IDs[node_ID]
    occupied_space, infer_num = tail_Space(node_ID, matcher.node_tail_length[node_ID], sym_ID, matcher, matchinginfo) 
    rest_target_length = length(subterms) - last_host_subterm_id - occupied_space

    inferred_length = rest_target_length / infer_num
    if isinteger(inferred_length)
        matchinginfo.matched_sym_nodes[sym_ID] = subterms[last_host_subterm_id .+ (1:Int(inferred_length))]
        return true, inferred_length # no practical need to check free term
    else
        return false, 0
    end
end

local_Length(::Union{FIXED_OP, IDPDT_OP, DPDT_OP, FIXED_SINGLE, IDPDT_SINGLE, DPDT_SINGLE}, node_ID::Integer, matcher::Matcher, matchinginfo::MatchingInfo) = 1
local_Length(::DPDT_FREE, node_ID::Integer, matcher::Matcher, matchinginfo::MatchingInfo) = length(get_matched_node(node_ID, matcher, matchinginfo))
local_Length(::Union{IDPDT_FREE, IDPDT_INFER}, node_ID::Integer, matcher::Matcher, matchinginfo::MatchingInfo) = 0
function tail_Space(node_ID::Integer, tail_length::Integer, sym_ID::Integer, matcher::Matcher, matchinginfo::MatchingInfo) 
    infer_num = 1
    occupied_space = 0
    for local_node_ID in ((1:tail_length) .+ node_ID)
        if matcher.node_sym_IDs[local_node_ID] == sym_ID
            infer_num += 1
        else
            occupied_space += local_Length(get_Tag(matcher.matcher_nodes[local_node_ID]), local_node_ID, matcher, matchinginfo)
        end
    end
    return occupied_space, infer_num
end

embody_GeneralTerm(this_node::GroundTerm, this_match::Dict) = deepcopy(this_node) #rewriting rule always generate new copies, e.g., in a => a + a, a2 and a3 should be different copies
function embody_GeneralTerm(this_node::SubtermVariable, this_match::Dict)
    if this_node.sym in keys(this_match)
        return deepcopy(this_match[this_node.sym])
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
    if this_node.tag isa FIXED_OP
        semantic_operation = syntax_operation
    else
        if syntax_operation in keys(this_match)
            semantic_operation = this_match[syntax_operation]
        elseif syntax_operation in keys(AUX_SYM_DEFINITION)
            this_func, syntactic_symbols = AUX_SYM_DEFINITION[syntax_operation]
            semantic_operation = [this_match[sym] for sym in syntactic_symbols]
        end
    end
    return construct_Term(semantic_operation, semantic_subterms)
end

function apply_Rules_One_Node(source_term::GroundTerm, this_rule::RewritingRule)
    @Takeout (matcher, matchinginfo, structure_to_produce) FROM this_rule
    is_match = match_Global(source_term, matcher, matchinginfo)
    return is_match ? (true, embody_GeneralTerm(structure_to_produce, Dict(matcher.syms .=> matchinginfo.matched_sym_nodes))) : (false, source_term)
end
function apply_Rules_One_Node(source_term::GroundTerm, rules::Vector{RewritingRule})
    term_changed, this_term = false, source_term
    for this_rule in rules
        this_check, this_term = apply_Rules_One_Node(this_term, this_rule)
        term_changed |= this_check
    end
    return term_changed, this_term
end

apply_Rules(source_term::Union{FEM_Float, SymbolicWord}, rules) = apply_Rules_One_Node(source_term, rules)
function apply_Rules(source_term::SymbolicTerm, rules)
    local_changed, this_term = true, source_term

    head_changed = false
    while local_changed
        local_changed, this_term = apply_Rules_One_Node(this_term, rules)
        head_changed |= local_changed
    end

    subterm_changed = false
    if this_term isa SymbolicTerm
        for i = 1:length(this_term.subterms)
            local_changed, this_term.subterms[i] = apply_Rules(this_term.subterms[i], rules)
            subterm_changed |= local_changed
        end
    end
    return (head_changed || subterm_changed), refresh_Term(this_term, subterm_changed)
end