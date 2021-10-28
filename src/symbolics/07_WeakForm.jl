function register_SymbolicTerm(input_ex::Expr, output_ex::Expr)
    lhs, rhs = input_ex.args
    this_term = construct_Term(rhs)

    if lhs isa Symbol
        term_name = lhs
        free_index = get_FreeIndex(this_term)
        ~isempty(free_index) && error("Scalar cannot have free indices")

        final_term = simplify_Symbolics(this_term)
        println("Scalar ", term_name, " declared as ", visualize(this_term), " them simplified to ", visualize(final_term))
        push!(output_ex.args, :($term_name = $final_term))
    elseif lhs.head == :curly && length(lhs.args) > 1 && lhs.args[2] isa Symbol
        term_name = lhs.args[1]
        declared_free_index = lhs.args[2:end]
        source_free_index = get_FreeIndex(this_term)
        isempty(symdiff(declared_free_index, source_free_index)) || error("Free indices must match", declared_free_index, source_free_index)

        final_term = simplify_Symbolics(this_term)
        println("Tensor ", term_name, " declared as ", visualize(this_term), " with free index ", join(string.(declared_free_index), ", "), " them simplified to ", visualize(final_term))
        push!(output_ex.args, :($term_name = $this_term))
    else
        println(lhs)
        println(rhs)
        error("Wrong grammar")
    end
    return output_ex
end

macro Def(input_ex)
    input_ex_batch = vectorize_Args(input_ex)
    output_ex = Expr(:block)
    for this_ex in input_ex_batch
        output_ex = this_ex.head == :(=) ? register_SymbolicTerm(this_ex, output_ex) : error("Wrong syntax")
    end
    return esc(output_ex)
end

function parse_BilinearForm(this_term::GroundTerm)
    this_term.subterms[1] isa SymbolicTerm && println(visualize(this_term.subterms[1]))
    Symbolic_BilinearForm(this_term.subterms...)
end

function parse_WeakForm(source_term::SymbolicTerm, dim::Integer)
    this_term = unroll_Dumb_Indices(source_term, dim) |> simplify_Mechanics |> simplify_Mechanics
    ~isempty(this_term.free_index) && error("WeakForm must be concrete", this_term)

    if this_term.operation == :+
        bilinear_forms = collect(parse_BilinearForm.(this_term.subterms))
    elseif this_term.operation == :Bilinear
        bilinear_forms = [parse_BilinearForm(this_term)]
    else
        error("Wrong syntax")
    end
    return Symbolic_WeakForm(filter(x -> x.base_term != 0, bilinear_forms))
end

