Base.isless(::Integer, ::Symbol) = true
Base.isless(::Symbol, ::Integer) = false
# Base.isless(::FEM_Int, ::Symbol) = true
# Base.isless(::Symbol, ::FEM_Int) = false
function construct_Word(base_variable::Symbol, c_ids::Vector, d_ids::Vector)
    if isempty(d_ids)
        return SymbolicWord(base_variable, FEM_Int(0), c_ids, IndexSym[])
    else
        total_d_ids_num = length(d_ids)
        filter!(x -> x != :t, d_ids)
        return SymbolicWord(base_variable, total_d_ids_num - length(d_ids), c_ids, sort!(d_ids))
    end
end

function parse_Word_Index(this_word::SymbolicWord)
    @Takeout (c_ids, sd_ids) FROM this_word
    total_ids = [c_ids; sd_ids]
    free_index, dumb_index = Symbol[], Symbol[]
    for arg in total_ids
        arg isa Number && continue
        :t in total_ids && error("t is not allowed in component index, or there is a bug and derivative index t is not processed")
        if arg in dumb_index
            error("$arg appears 3 times")
        elseif arg in free_index
            push!(dumb_index, arg)
        else
            push!(free_index, arg)
        end
    end
    setdiff!(free_index, dumb_index)
    return free_index, dumb_index
end

const VOIGT_INDEX_2D = FEM_Int[1 3; 3 2]
const VOIGT_INDEX_3D = FEM_Int[1 6 5; 6 2 4; 5 4 3]
const INVERSE_VOIGT_INDEX_2D = Tuple{FEM_Int, FEM_Int}[(1, 1), (2, 2), (1, 2)]
const INVERSE_VOIGT_INDEX_3D = Tuple{FEM_Int, FEM_Int}[(1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2)]
function Voigt_ID(i, j, dim)
    if dim == 2
        return VOIGT_INDEX_2D[i,j]
    elseif dim == 3
        return VOIGT_INDEX_3D[i,j]
    else
        error("Undefined symmetry")
    end
end
function inverse_Voigt_ID(i, dim)
    if dim == 2
        return INVERSE_VOIGT_INDEX_2D[i]
    elseif dim == 3
        return INVERSE_VOIGT_INDEX_3D[i]
    else
        error("Undefined symmetry")
    end
end

function word_To_Sym(dim::Integer, base_variable::Symbol, td_order::Integer, c_ids::Vector, sd_ids::Vector)
    full_name ="$base_variable"
    if length(c_ids) == 0
    elseif length(c_ids) == 1
        full_name = "$(full_name)$(c_ids[1])"
    else
        if :SYMMETRIC_TENSOR in get_VarAttribute(base_variable)
            full_name = "$(full_name)$(Voigt_ID(c_ids..., dim))"
        else
            ID = 1 + sum([(c_id - 1) * dim ^ (i - 1) for (i, c_id) in enumerate(c_ids)])
            full_name = "$(full_name)$(ID)"
        end
    end

    if td_order > 0
        full_name = "$(full_name)_$(repeat("t", td_order))"
    end

    if ~(isempty(sd_ids))
        full_name = "$(full_name)_$(sd_ids...)"
    end
    return Symbol(full_name)
end

get_VarAttribute(sym::Symbol) = get(VARIABLE_ATTRIBUTES, sym, Symbol[])
get_VarAttribute(word::SymbolicWord) = get_VarAttribute(word.base_variable)
# word_To_SymType(symbolic_word::SymbolicWord) = symbolic_word.base_variable
word_To_BaseSym(dim::Integer, symbolic_word::SymbolicWord) = word_To_Sym(dim, symbolic_word.base_variable, FEM_Int(0), symbolic_word.c_ids, IndexSym[])
word_To_LocalSym(dim::Integer, symbolic_word::SymbolicWord) = word_To_Sym(dim, symbolic_word.base_variable, symbolic_word.td_order, symbolic_word.c_ids, IndexSym[])
word_To_TotalSym(dim::Integer, symbolic_word::SymbolicWord) = word_To_Sym(dim, symbolic_word.base_variable, symbolic_word.td_order, symbolic_word.c_ids, symbolic_word.sd_ids)