visualize(x::Union{Number, Symbol}) = "$x"

const SUBSCRIPT_MAPPING = Dict(:i => "ᵢ", :j => "ⱼ", :k => "ₖ", :l => "ₗ", :m => "ₘ", :n => "ₙ", :o => "ₒ", :p => "ₚ", :t => "ₜ", 1 => "₁", 2 => "₂", 3 => "₃")
decorate_Subscript(x) = get(SUBSCRIPT_MAPPING, x, "ₓ") 
function update_Symbolic_Word_Name(base_variable::Symbol, td_order::Integer, c_ids, sd_ids)
    full_name = "$(base_variable)$(join(decorate_Subscript.(c_ids)))"
    full_name = td_order > 0 ? "$(full_name),$(repeat("ₜ", td_order))" : full_name
    full_name = isempty(sd_ids) ? full_name : "$(full_name),$(join(decorate_Subscript.(sd_ids)))"
    return full_name
end

"""
    visualize(x::Union{SymbolicWord, SymbolicTerm, SubtermVariable, FunctionVariable, RewritingRule, Symbolic_BilinearForm, Vector{Symbolic_BilinearForm}})

Print the expressions.
"""
visualize(x::SymbolicWord) = update_Symbolic_Word_Name(x.base_variable, x.td_order, x.c_ids, x.sd_ids)

function update_Term_Name(operation::Symbol, substrings)
    if operation in (:(+), :(-), :(*), :(/), :(^)) && length(substrings) > 1
        head, sep, tail = "(", " $operation ", ")"
    else
        head, sep, tail = "$operation(", ", ", ")"
    end
    body = join(substrings, sep)
    return string(head, body, tail)
end
visualize(x::SymbolicTerm) = update_Term_Name(x.operation, visualize.(x.subterms))

visualize(x::SubtermVariable) = "$(x.sym)"
visualize(x::FunctionVariable) =  update_Term_Name(x.operation, visualize.(x.subterms))
visualize(x::RewritingRule) = "$(visualize(x.structure_to_match)) => $(visualize(x.structure_to_produce))"

function get_TensorName(x::TensorInfo) 
    base_sym, _, td_order, sd_order = x
    full_name = "$base_sym"
    if td_order > 0
        full_name = "$(full_name)_$(repeat("t", td_order))"
    end
    if sd_order > 0
        full_name = "$(full_name)_$(repeat("x", sd_order))"
    end
    return full_name
end
get_DiffSym(x::TensorInfo, y::TensorInfo) = Symbol("d($(get_TensorName(x)), $(get_TensorName(y)))")
visualize(x::PhysicalTensor) = "$(get_TensorName(x.tensor_info)) = $(visualize(x.definition))"

visualize(x::Symbolic_BilinearForm) = "($(visualize(x.dual_word)), $(visualize(x.base_term)))"

Base.show(io::IO, x::Union{SymbolicWord, SymbolicTerm, SubtermVariable, FunctionVariable, RewritingRule, PhysicalTensor, Symbolic_BilinearForm}) = print(io, visualize(x))
# note here performance is needed since it will also be used in rewriting, so do not consider dumb indices
Base.hash(x::SymbolicWord, h::UInt) = hash([x.base_variable, x.td_order, x.c_ids, x.sd_ids], h)
Base.:(==)(x::SymbolicWord, y::SymbolicWord) = (x.base_variable == y.base_variable) && (x.td_order == y.td_order) && (x.c_ids == y.c_ids) && (x.sd_ids == y.sd_ids)

Base.hash(x::SymbolicTerm, h::UInt) = hash(x.subterms, hash(x.operation, h))
Base.:(==)(x::SymbolicTerm, y::SymbolicTerm) = (x.operation == y.operation) && (x.subterms == y.subterms)