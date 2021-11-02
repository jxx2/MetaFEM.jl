# construct_Word(arg::Symbol) = SymbolicWord(arg, FEM_Int(0), (), ())
function construct_Word(base_variable::Symbol, c_ids::Vector, d_ids::Vector)
    if isempty(d_ids)
        td_order, sd_ids = FEM_Int(0), IndexSym[]
    else
        time_derivative_ids = d_ids .== :t
        td_order = sum(time_derivative_ids)
        sd_ids = d_ids[.~time_derivative_ids]
    end
    sorted_sd_ids = sd_ids[sortperm(string.(sd_ids))]
    return SymbolicWord(base_variable, td_order, Tuple(c_ids), Tuple(sorted_sd_ids))
end

function parse_Word_Index(this_word::SymbolicWord)
    @Takeout (c_ids, sd_ids) FROM this_word
    free_index, dumb_index = Dict{Symbol, Tuple{FEM_Int, FEM_Int}}(), Dict{Symbol, Tuple{FEM_Int, FEM_Int, FEM_Int, FEM_Int}}()
    for (arg_pos, arg) in enumerate(c_ids)
        arg isa Number && continue
        :t in c_ids && error("t is not allowed in component index")
        if arg in keys(free_index)
            dumb_index[arg] = tuple(free_index[arg]..., FEM_Int(0), arg_pos)
            delete!(free_index, arg)
        elseif arg in keys(dumb_index)
            error(arg, "appears 3 times")
        else
            free_index[arg] = (FEM_Int(0), arg_pos)
        end
    end
    for (arg_pos, arg) in enumerate(sd_ids)
        arg isa Number && continue
        arg == :t && error("t is not allowed in spatial derivative index")
        if arg in keys(free_index)
            dumb_index[arg] = tuple(free_index[arg]..., FEM_Int(1), arg_pos)
            delete!(free_index, arg)
        elseif arg in keys(dumb_index)
            error(arg, "appears 3 times")
        else
            free_index[arg] = (FEM_Int(1), arg_pos)
        end
    end
    return free_index, dumb_index
end

VOIGT_INDEX_2D = Dict([(i, i) => i for i = 1:2]..., [(i, j) => 3 for i = 1:2, j = 1:2 if i != j]...) 
VOIGT_INDEX_3D = Dict([(i, i) => i for i = 1:3]..., (2, 3) => 4, (3, 2) => 4, (3, 1) => 5, (1, 3) => 5, (1, 2) => 6, (2, 1) => 6) 
asymmetric_Index(c_ids, dim::Integer) = sum(collect(c_ids) .* [dim ^ (i - 1) for i = 1: length(c_ids)])

function word_To_Sym(dim::Integer, base_variable::Symbol, td_order::Integer, c_ids::Tuple, sd_ids::Tuple)
    full_name_container = Union{Number, Symbol, String, Char}[base_variable]
    if length(c_ids) == 0
    elseif length(c_ids) == 1
        c_ids[1] isa Number || error("should be a number")
        push!(full_name_container, c_ids[1])
    else
        if :SYMMETRIC_TENSOR in VARIABLE_ATTRIBUTES[base_variable]
            if dim == 2
                push!(full_name_container, VOIGT_INDEX_2D[c_ids])
            elseif dim == 3
                push!(full_name_container, VOIGT_INDEX_3D[c_ids])
            end
        else
            push!(full_name_container, asymmetric_Index(c_ids, dim))
        end
    end

    if td_order > 0
        push!(full_name_container, "_")
        append!(full_name_container, ["t" for i = 1:td_order])
    end

    if ~(isempty(sd_ids))
        push!(full_name_container, "_")
        append!(full_name_container, sd_ids)
    end
    return Symbol(string(full_name_container...))
end

word_To_SymType(symbolic_word::SymbolicWord) = symbolic_word.base_variable
word_To_BaseSym(dim::Integer, symbolic_word::SymbolicWord) = word_To_Sym(dim, symbolic_word.base_variable, FEM_Int(0), symbolic_word.c_ids, ())
word_To_LocalSym(dim::Integer, symbolic_word::SymbolicWord) = word_To_Sym(dim, symbolic_word.base_variable, symbolic_word.td_order, symbolic_word.c_ids, ())
word_To_TotalSym(dim::Integer, symbolic_word::SymbolicWord) = word_To_Sym(dim, symbolic_word.base_variable, symbolic_word.td_order, symbolic_word.c_ids, symbolic_word.sd_ids)

