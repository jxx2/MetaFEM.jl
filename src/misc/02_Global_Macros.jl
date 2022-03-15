using Base: String
function vectorize_Args(arg)
    if is_block(arg)
        return stripblocks(striplines(arg)).args[:]
    elseif is_collection(arg) #$arg isa Expr && $arg.head in [:tuple; :vect; :vcat]
        return arg.args[:]
    elseif arg isa Tuple
        return collect(arg)
    else
        return [arg]
    end
end

macro If_Match(sentence, syntax_structure...)
    sentence_length = length(syntax_structure) - 1 #the last one is the command

    keys = Pair[]
    term_conditions = Expr[]
    for (index, arg) in enumerate(syntax_structure)
        if arg isa Expr && arg.head == :vect
            push!(keys, index => arg.args[1])
        elseif arg isa Symbol
            push!(term_conditions, :($sentence[$index] == $(Meta.quot(arg))))
        end
    end

    condition = :(length($sentence) == $sentence_length)
    for this_condition in term_conditions
        condition = :($condition && $this_condition)
    end
    content = Expr(:block)
    for (index, name) in keys
        push!(content.args, :($name = $sentence[$index]))
    end
    push!(content.args, syntax_structure[end])
    ex =
    :(if $condition
         $content
      end)
    return esc(ex)
end

"""
    @Takeout a, b FROM c

equals to

    a = c.a
    b = c.b

while

    @Takeout a, b FROM c WITH PREFIX d

equals to

    da = c.a
    db = c.b
"""
macro Takeout(sentence...)
    ex = Expr(:block)
    arg_tuple = vectorize_Args(sentence[1])
    @If_Match sentence[2:3] FROM [host] for arg in arg_tuple
        argchain = Symbol[]
        this_arg = arg
        while this_arg isa Expr && arg.head == :.
            push!(argchain, this_arg.args[2].value)
            this_arg = this_arg.args[1]
        end
        push!(argchain, this_arg)
        reverse!(argchain)
        target_arg = argchain[end]
        source_arg = host
        for this_arg in argchain
            source_arg = Expr(:., source_arg, QuoteNode(this_arg))
        end
        push!(ex.args, :($target_arg = $source_arg))
    end

    @If_Match sentence[4:end] WITH PREFIX [prefix] for arg in ex.args
        arg.args[1] = Symbol(string(prefix, arg.args[1]))
    end

    return esc(ex)
end

Base.:|>(x1, x2, f) = f(x1, x2) # (a, b)... |> f = f(a, b)
Base.:|>(x1, x2, x3, f) = f(x1, x2, x3)
Base.:|>(x1, x2, x3, x4, f) = f(x1, x2, x3, x4)

reload_Pipe(x) = x
function reload_Pipe(x::Expr)
    (isexpr(x, :call) && x.args[1] == :|>) || return x
    xs, (node_func, extra_args...) = vectorize_Args.(x.args[2:3])
    is_splated = isexpr(node_func, :...)
    
    if is_splated
        node_func = node_func.args[1]
    end

    res = isexpr(node_func, :call) ? Expr(node_func.head, node_func.args[1], xs..., node_func.args[2:end]...) : Expr(:call, node_func, xs...)

    if is_splated
        res = Expr(:..., res)
    end
    return Expr(:tuple, res, extra_args...)
end

macro Pipe(exs)
    esc(postwalk(reload_Pipe, exs).args[1])
end

macro Construct(typename) 
    if typename isa Symbol
        target_type = eval(typename)
    elseif typename.head == :curly
        target_type = eval(typename.args[1])
    end
    fields = fieldnames(target_type)
    return esc(:($typename($(fields...))))
end

const FEM_Int = Int32 # Note, hashing some GPU FEM_Int like indices may assume last 30 bits in 2D, 20 bits in 3D, so not really needed anything above Int32
const FEM_Float = Float64 # Double digit make iterative solvers much more robust, since we are still simply Jacobi conditioned
VTuple(T...) = Tuple{Vararg{T...}}

const _UNITS = Dict{String, Int}(["B", "KB", "MB", "GB", "TB"] .=> [Int(1 << (10 * i)) for i = 0:4])
mutable struct MemUnit
    u_name::String
    u_size::Int
end

function (_obj::MemUnit)(unit_name::String)
    if unit_name in keys(_UNITS)
        _obj.u_name = unit_name
        _obj.u_size = _UNITS[unit_name]
    end
end

"""
    MEM_UNIT(unit_name::String)

Set the displayed memory unit where `unit_name` can be "B", "KB", "MB", "GB", "TB".
"""
const MEM_UNIT = MemUnit("MB", _UNITS["MB"])

function estimate_msize(x::T, budget::Integer, counter_func::Function) where T
    msize = counter_func(x, budget)
    if isa(x, Dict)
        for dict_val in values(x)
            msize += estimate_msize(dict_val, budget, counter_func)
        end
    elseif isa(x, AbstractArray)

    elseif isstructtype(T) # T can't be an abstract array to avoid calculating specific arrays twice, e.g., CuArray
        for fname in fieldnames(T)
            msize += estimate_msize(getfield(x, fname), budget, counter_func)
        end
    end
    return msize
end # recursively count the total size of an object

CPU_sizeof(x::T, budget::Integer = 0) where T <: AbstractArray = sizeof(x) + Base.elsize(T) * budget
CPU_sizeof(x, budget::Integer = 0) = sizeof(x)
GPU_sizeof(x::T, budget::Integer = 0) where T <: CuArray = sizeof(x) + Base.elsize(T) * budget
GPU_sizeof(x, budget::Integer = 0) = 0 

estimate_memory_CPU(x, budget::Integer = 0) = estimate_msize(x, budget, CPU_sizeof) 
estimate_memory_GPU(x, budget::Integer = 0) = estimate_msize(x, budget, GPU_sizeof)

report_memory(x, _obj::MemUnit) where N = string("$(estimate_memory_CPU(x)/ _obj.u_size) $(_obj.u_name) CPU memory and $(estimate_memory_GPU(x)/ _obj.u_size) $(_obj.u_name) GPU memory")
report_memory(x) = report_memory(x, MEM_UNIT)
# Slightly prettier gensym
mutable struct PrefixGenerator
    prefix::String
    ID::Int
    PrefixGenerator(prefix::String) = new(prefix, 0)
end
(_obj::PrefixGenerator)(x = "") = Symbol("$(_obj.prefix)_$(_obj.ID+=1)_$x")