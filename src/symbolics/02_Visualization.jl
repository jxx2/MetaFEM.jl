visualize(x::Union{Number, Symbol}) = string(x)

SUBSCRIPT_MAPPING = Dict(:i => "ᵢ", :j => "ⱼ", :k => "ₖ", :l => "ₗ", :m => "ₘ", :n => "ₙ",
 :o => "ₒ", :p => "ₚ", :t => "ₜ", 0 => "ₜ", 1 => "₁", 2 => "₂", 3 => "₃")
decorate_Subscript(x) = get(SUBSCRIPT_MAPPING, x, "₀")
function update_Symbolic_Word_Name(base_variable::Symbol, td_order::Integer, c_ids::Union{Tuple, Vector}, sd_ids::Union{Tuple, Vector})
    full_name_container = Union{Number, Symbol, String, Char}[]
    push!(full_name_container, base_variable)
    append!(full_name_container, decorate_Subscript.(c_ids))

    if td_order > 0
        push!(full_name_container, ",")
        append!(full_name_container, ["ₜ" for i = 1:td_order])
    end

    if ~isempty(sd_ids)
        push!(full_name_container, ",")
        append!(full_name_container, decorate_Subscript.(sd_ids))
    end
    return string(full_name_container...)
end

"""
    visualize(x::Union{SymbolicWord, SymbolicTerm, SubtermVariable, FunctionVariable, RewritingRule, Symbolic_BilinearForm, Symbolic_WeakForm})

Print the expressions.
"""
visualize(x::SymbolicWord) =  update_Symbolic_Word_Name(x.base_variable, x.td_order, x.c_ids, x.sd_ids)

function update_Term_Name(operation::Symbol, substrings::Vector)
    if operation in (:(+), :(-), :(*), :(/), :(^)) && length(substrings) > 1
        head, sep, tail = "(", string(" ", operation, " "), ")"
    else
        head, sep, tail = string(operation, "("), ", ", ")"
    end
    body = join(substrings, sep)
    full_name = string(head, body, tail)
    return full_name
end
visualize(x::SymbolicTerm) = update_Term_Name(x.operation, collect(visualize.(x.subterms)))

visualize(x::SubtermVariable) = string(x.sym)
visualize(x::FunctionVariable) =  update_Term_Name(x.operation, collect(visualize.(x.subterms)))
visualize(x::RewritingRule) = string(visualize(x.structure_to_match), " => ", visualize(x.structure_to_produce))
visualize(x::Symbolic_BilinearForm) = string("(", visualize(x.dual_term), ", ", visualize(x.base_term), ")")
visualize(x::Symbolic_WeakForm) = join(visualize.(x.bilinear_forms), " + ")

Base.show(io::IO, x::Union{SymbolicWord, SymbolicTerm, SubtermVariable, FunctionVariable, RewritingRule, Symbolic_BilinearForm, Symbolic_WeakForm}) = 
print(io, visualize(x))