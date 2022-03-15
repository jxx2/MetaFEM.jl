using Base: AbstractFloat
mutable struct Polynomial{dim}
    factors::Vector{FEM_Float}
    orders::Vector{Tuple{Vararg{FEM_Int, dim}}}

    Polynomial(dim::Integer) = new{dim}([FEM_Float(0.)], [const_Tup(FEM_Int(0), dim)])
    Polynomial(factor::AbstractFloat, order::Tuple) = new{length(order)}(FEM_Float[factor], [FEM_Int.(order)])
    Polynomial(factors::Vector, orders::Vector) = new{length(orders[1])}(FEM_Float.(factors), map(x -> FEM_Int.(x), orders))
    Polynomial(example::Polynomial{dim}) where dim = new{dim}(copy(example.factors), copy(example.orders))
end

const_Tup(w, dim_num) = Tuple(fill(w, dim_num))
function basis_Tup(dim_id, dim_num, wi = 1, wo = 0)
    temp = fill(wo, dim_num)
    temp[dim_id] = wi
    return Tuple(temp)
end
collect_Basis(dim::Integer) = [Polynomial(1., basis_Tup(i, dim)) for i = 1:dim]

function Base.:+(p1::Polynomial{dim}, num::Number) where dim
    p_ans = Polynomial(p1)
    num == 0. && return p_ans

    this_order = const_Tup(FEM_Int(0), dim)
    matched_id = findfirst(x -> x == this_order, p1.orders)
    if isnothing(matched_id)
        push!(p_ans.factors, num)
        push!(p_ans.orders, this_order)
    else
        p_ans.factors[matched_id] += num
        p_ans = check_Clear(p_ans)
    end
    return p_ans
end

Base.:+(num::Number, p1::Polynomial{dim}) where dim = p1 + num

function Base.:+(p1::Polynomial{dim}, p2::Polynomial{dim}) where dim
    p_ans = Polynomial(p1)
    for (this_factor, this_order) in zip(p2.factors, p2.orders)
        matched_id = findfirst(x -> x == this_order, p1.orders)
        if isnothing(matched_id)
            push!(p_ans.factors, this_factor)
            push!(p_ans.orders, this_order)
        else
            p_ans.factors[matched_id] += this_factor
        end
    end
    return check_Clear(p_ans)
end

function Base.:-(p1::Polynomial{dim}) where dim
    p_ans = Polynomial(p1)
    p_ans.factors .*= -1
    return p_ans
end
Base.:-(p1::Polynomial{dim}, num::Number) where {dim} = p1 + (-num)
Base.:-(num::Number, p1::Polynomial{dim}) where {dim} = -p1 + num
Base.:-(p1::Polynomial{dim}, p2::Polynomial{dim}) where {dim} = p1 + (-p2)

function check_Clear(p1::Polynomial{dim}) where dim
    id_to_remove = findall(x -> x == 0., p1.factors)
    # not_empty = p1.factors .!= 0
    err = 1e-8
    not_empty = abs.(p1.factors) .>= err
    if sum(not_empty) == 0
        p1.factors = [FEM_Float(0.)]
        p1.orders = [const_Tup(FEM_Int(0), dim)]
    else
        p1.factors = p1.factors[not_empty]
        p1.orders = p1.orders[not_empty]
    end
    return p1
end

function Base.:*(p1::Polynomial{dim}, num::Number) where dim
    p_ans = Polynomial(p1)
    if num == 0
        p_ans.factors = [FEM_Float(0.)]
        p_ans.orders = [const_Tup(FEM_Int(0), dim)]
    else
        p_ans.factors .*= num
    end
    return p_ans
end

Base.:*(num::Number, p1::Polynomial{dim}) where dim = p1 * num
Base.:/(p1::Polynomial{dim}, num::Number) where dim = p1 * (1/num)

function Base.:*(p1::Polynomial{dim}, p2::Polynomial{dim}) where dim
    p_ans = Polynomial(dim)
    for (first_factor, first_order) in zip(p1.factors, p1.orders)
        for (second_factor, second_order) in zip(p2.factors, p2.orders)
            this_factor = first_factor * second_factor
            this_order = first_order .+ second_order

            matched_id = findfirst(x -> x == this_order, p_ans.orders)
            if isnothing(matched_id)
                push!(p_ans.factors, this_factor)
                push!(p_ans.orders, this_order)
            else
                p_ans.factors[matched_id] += this_factor
            end
        end
    end
    return check_Clear(p_ans)
end

function Base.:^(p1::Polynomial{dim}, num::Integer) where dim #can be rewrite in to a ^ 4 = (a^2)^2 ...
    p_ans = Polynomial(FEM_Float(1.), const_Tup(FEM_Int(0), dim))
    for i = 1:num
        p_ans *= p1
    end
    return p_ans
end

function substitute_Polynomial(p_src::Polynomial{dim1}, src_dim::Integer, p_template::Polynomial{dim2}) where {dim1, dim2}
    p_ans = Polynomial(dim2)
    for (this_factor, p_src_order) in zip(p_src.factors, p_src.orders)
        p_core = p_template ^ p_src_order[src_dim]
        p_base = Polynomial(this_factor, ntuple(x -> ((x == src_dim) || (x > dim1)) ? 0 : p_src_order[x], dim2))
        p_ans += p_core * p_base        
    end
    return p_ans
end

function derivative(p1::Polynomial{dim}, orders::Tuple) where dim
    p2 = Polynomial(p1)
    for i = 1:length(p2.factors)
        this_order = p2.orders[i]
        if minimum(this_order.- orders) < 0
            p2.factors[i] = FEM_Int(0)
            continue
        end
        for this_dim = 1:dim
            p2.factors[i] *= factorial(this_order[this_dim])/factorial(this_order[this_dim] - orders[this_dim])
        end
        p2.orders[i] = this_order .- orders
    end
    return check_Clear(p2)
end

function evaluate_Polynomial(p1::Polynomial{dim}, pos::Tuple) where dim
    sum = 0
    for (this_factor, this_order) in zip(p1.factors, p1.orders)
        this_term = prod(pos .^ this_order)
        sum += this_term * this_factor
    end
    return sum
end