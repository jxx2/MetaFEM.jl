VARIABLE_ATTRIBUTES = Dict{Symbol, Tuple}()
function enum_Local(variable_set, attr::Vector{Symbol} = Symbol[])
    ex = Expr(:block)
    for sub_arg in variable_set
        arg_tuple = vectorize_Args(sub_arg)
        sym = arg_tuple[1]
        param = arg_tuple[2:end]

        push!(ex.args, :(
        begin
            VARIABLE_ATTRIBUTES[$(Meta.quot(sym))] = $(param..., attr...)
            println($(Meta.quot(sym)), " is declared")
        end))
    end
    return ex
end

macro Sym(variable_set...) 
    return esc(enum_Local(variable_set))
end

macro External_Sym(variable_set...)
    return esc(enum_Local(variable_set, [:EXTERNAL_VAR]))
end

@External_Sym (x, CONTROLPOINT_VAR) (y, CONTROLPOINT_VAR) (z, CONTROLPOINT_VAR) (t, GLOBAL_VAR) (dt, GLOBAL_VAR)
@External_Sym (F, INTEGRATION_POINT_VAR) (f, INTEGRATION_POINT_VAR) (n, INTEGRATION_POINT_VAR) δ
@Sym u p T

const IndexSym = Union{FEM_Int, Symbol}
struct SymbolicWord #Word (the prefix symbolic is to just make it parallel to symbolic term)
    base_variable::Symbol
    td_order::FEM_Int
    c_ids::Tuple{Vararg{IndexSym}}
    sd_ids::Tuple{Vararg{IndexSym}}
end

struct SymbolicTerm
    operation::Symbol
    subterms::Tuple #Union{Number, SymbolicWord, SymbolicTerm}
    free_index::Tuple{Vararg{Symbol}} #The free index this SymbolicTerm holds
    dumb_index::Tuple{Vararg{Symbol}} #The dumb index this SymbolicTerm holds
end

const GroundTerm = Union{FEM_Float, SymbolicWord, SymbolicTerm}
# const SYMBOL_TO_GROUND_TERM = Dict{Symbol, Union{FEM_Float, SymbolicWord, SymbolicTerm, Tuple}}()

struct FunctionVariable 
    operation::Symbol
    operation_is_fixed::Bool
    subterms::Tuple
end

abstract type SubtermVariable end #Node is the syntactic variable in theory (variable is not a good name in code)
struct FixSubtermVariable <: SubtermVariable
    sym::Symbol
end
struct FreeSubtermVariable <: SubtermVariable
    sym::Symbol
end
GeneralTerm = Union{FunctionVariable, SubtermVariable, GroundTerm}

struct Matcher
    match_nodes::Vector{GeneralTerm} #list of sym for each node
    parent_node_IDs::Vector{FEM_Int}

    syms::Vector{Symbol} #list of syms to match
    nodes_2_syms::Vector{FEM_Int} #list of sym id in syms for each node

    is_independent::BitArray #if this node is the first time of this sym then yes
    size_inferences::Vector{Tuple{Bool, FEM_Int, FEM_Int, Vector{FEM_Int}}} #size_inferences, numbers of fixed, numbers of self, IDs of dependence
end

struct RewritingRule
    structure_to_match::GeneralTerm
    structure_to_produce::GeneralTerm

    matcher::Matcher
end

struct Symbolic_BilinearForm #inner product
    dual_term::SymbolicWord #test term can only be a single term
    base_term::GroundTerm
end

struct Symbolic_WeakForm
    bilinear_forms::Vector{Symbolic_BilinearForm}
end