VARIABLE_ATTRIBUTES = Dict{Symbol, Vector{Symbol}}()
function enum_Local(variable_set, attr::Vector{Symbol} = Symbol[])
    ex = Expr(:block)
    for sub_arg in variable_set
        arg_tuple = vectorize_Args(sub_arg)
        sym = arg_tuple[1]
        param = arg_tuple[2:end]

        push!(ex.args, :(
        begin
            VARIABLE_ATTRIBUTES[$(Meta.quot(sym))] = $(Symbol[param..., attr...])
            println($(Meta.quot(sym)), " is declared")
        end))
    end
    return ex
end

"""
    @Sym (name, attribute_1, attribute_2,...)
    @External_Sym (name, attribute_1, attribute_2,...)

`@Sym` declares a variable symbol `name` with the attributes `attribute_1`, `attribute_2`, ..., 
which is simply stored as the pair `name` => (`:INTERNAL_VAR`, `attribute_1`, `attribute_2`, ...) in the exposed global dictionary `VARIABLE_ATTRIBUTES`.

`@External_Sym` is just the `@Sym` with the default attribute: `:EXTERNAL_VAR` instead of `:INTERNAL_VAR`.
"""
macro Sym(variable_set...) 
    return esc(enum_Local(variable_set, [:INTERNAL_VAR]))
end

macro External_Sym(variable_set...)
    return esc(enum_Local(variable_set, [:EXTERNAL_VAR]))
end

const IndexSym = Union{FEM_Int, Symbol}
mutable struct SymbolicWord #Word (the prefix symbolic is to just make it parallel to symbolic term), word do not store free/dumb index and need to be rechecked everytime
    base_variable::Symbol
    td_order::FEM_Int
    c_ids::Vector{IndexSym}
    sd_ids::Vector{IndexSym}
    SymbolicWord(base_variable::Symbol) = new(base_variable, FEM_Int(0), IndexSym[], IndexSym[])
    function SymbolicWord(base_variable, td_order, c_ids, sd_ids) 
        if length(c_ids) == 2
            (:SYMMETRIC_TENSOR in get_VarAttribute(base_variable)) && sort!(c_ids)
        end
        new(base_variable, td_order, c_ids, sd_ids)
    end
end

mutable struct SymbolicTerm # each term only belongs to one parent
    operation::Symbol
    subterms::Vector #Union{Number, SymbolicWord, SymbolicTerm}
    free_index::Vector{Symbol} #The free index this SymbolicTerm holds
    dumb_index::Vector{Symbol} #The dumb index this SymbolicTerm holds
    SymbolicTerm(operation, subterms, free_index, dumb_index) = new(operation, convert(Vector{GroundTerm}, subterms), free_index, dumb_index)
end

const GroundTerm = Union{FEM_Float, SymbolicWord, SymbolicTerm}
mutable struct FunctionVariable
    operation::Symbol
    tag::Val
    subterms::Vector
end

mutable struct SubtermVariable #Node is the syntactic variable in theory (variable is not a good name in code)
    sym::Symbol
    tag::Val
end
GeneralTerm = Union{FunctionVariable, SubtermVariable, GroundTerm}

# op-node: fixed operation, idenpendent operation, dependent operation
# fixed single term (no tagged), 
# subterm-node: independent_single, dependent_single term, independent_free term, dependent_free term, independent_inferrable term dependent_inferrable term
const FIXED_OP, IDPDT_OP, DPDT_OP, FIXED_SINGLE, IDPDT_SINGLE, DPDT_SINGLE, IDPDT_FREE, DPDT_FREE, IDPDT_INFER = [Val{i} for i = 0:8]

struct Matcher # just a simple DFS (with subnodes eagerly expanded) 
    matcher_nodes::VTuple(GeneralTerm)

    node_sym_IDs::VTuple(Int8)
    node_parent_IDs::VTuple(Int8)
    node_tail_length::VTuple(Int8) # if 0, then the end of local subterms

    syms::VTuple(Symbol)
    sym_constraints::VTuple(Function)
end

mutable struct MatchingInfo
    matched_parent_nodes::Vector{SymbolicTerm}
    matched_sym_nodes::Vector{Union{Symbol, GroundTerm, Vector}} # store for each sym, may be symbol (as ground term operation), ground term or vector of ground term for function, normal variable, or free variable

    current_branch::FEM_Int
    branching_infos::Array{FEM_Int, 2} # node ID, start pos of target, size, 3xn
end

struct RewritingRule
    structure_to_match::GeneralTerm
    structure_to_produce::GeneralTerm

    matcher::Matcher
    matchinginfo::MatchingInfo # currently only one storage for single thread, will be expand-ed later
end

DEFINITION_TABLE = Dict{Symbol, Tuple{Vector{Symbol}, GroundTerm}}() # intermediate variables and integration point variables, still not loop, but if else

TensorInfo = Tuple{Symbol, FEM_Int, FEM_Int, FEM_Int} # base_sym, td_order, sd_order
mutable struct PhysicalTensor
    tensor_info::TensorInfo
    definition::GroundTerm
    free_index::Vector{Symbol}

    indexed_instances::Dict{Vector, GroundTerm}
end

mutable struct TensorTable
    dim::FEM_Int
    tensors::Dict{TensorInfo, PhysicalTensor}
    diff_tensors::Dict{Tuple{TensorInfo, TensorInfo}, TensorInfo}
    TensorTable(dim::Integer) = new(FEM_Int(dim), Dict{TensorInfo, PhysicalTensor}(), Dict{Tuple{TensorInfo, TensorInfo}, TensorInfo}())
end

struct Symbolic_BilinearForm #inner product
    dual_word::SymbolicWord #test term can only be a single word with fundamental symbol as the base_variable
    base_term::GroundTerm
end

function initialize_Definitions!()
    empty!(VARIABLE_ATTRIBUTES)
    empty!(DEFINITION_TABLE)

    @External_Sym (x, CONTROLPOINT_VAR) (y, CONTROLPOINT_VAR) (z, CONTROLPOINT_VAR) (t, GLOBAL_VAR) (dt, GLOBAL_VAR)
    @External_Sym (n, INTEGRATION_POINT_VAR) (δ, SYMMETRIC_TENSOR) ϵ
end
initialize_Definitions!()