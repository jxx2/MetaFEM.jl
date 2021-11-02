# 1  2  3  4
#############################
function init_Interpolation_Lagrange_1D(interp_order::Integer)
    interp_funcs_1D = Polynomial{1}[]
    interp_pos_1D = [i/interp_order for i = 0:interp_order]
    for this_interp_id = 1:(interp_order + 1)
        this_interp_pos = interp_pos_1D[this_interp_id]
        this_interp =  Polynomial(1) + 1
        for this_term_id = 1:(interp_order + 1)
            this_term_id == this_interp_id && continue

            this_term_pos = interp_pos_1D[this_term_id]
            denominator = this_interp_pos - this_term_pos
            #this term is just x
            factors = [1., -this_term_pos]./denominator
            orders = [(1,), (0,)]
            this_term = Polynomial(factors, orders)
            this_interp *= this_term
        end
        push!(interp_funcs_1D, this_interp)
    end
    return interp_funcs_1D
end

#13 14 15 16
# 9 10 11 12
# 5  6  7  8
# 1  2  3  4
#############################
function init_Interpolation_Cube_Lagrange(interp_order::Integer, dim::Integer)
    interp_funcs_1D = init_Interpolation_Lagrange_1D(interp_order)
    template_funcs = [substitute_Polynomial.(interp_funcs_1D, 1, Ref(Polynomial(1., ntuple(x -> x == i ? 1 : 0, dim)))) for i = 1:dim]

    result_itp_funcs = Polynomial{dim}[]
    for interp_ids in Iterators.product([1:(interp_order + 1) for i = 1:dim]...)
        push!(result_itp_funcs, prod([template_funcs[i][interp_ids[i]] for i = 1:dim]))
    end
    return result_itp_funcs
end

# 10
# 8  9 
# 5  6  7  
# 1  2  3  4
#############################
function init_Interpolation_Simplex_Lagrange(interp_order::Integer, dim::Integer)
    interp_funcs_1D = [Polynomial(1., (0,)), 
    [substitute_Polynomial(init_Interpolation_Lagrange_1D(this_order)[end], 1, Polynomial(interp_order / this_order, (1,))) for this_order = 1:interp_order]...]

    volumetric_coors = [Polynomial(1., ntuple(x -> x == i ? 1 : 0, dim)) for i = 1:dim]
    push!(volumetric_coors, Polynomial([1., [-1. for i = 1:dim]...], [ntuple(x -> 0, dim), [ntuple(x -> x == i ? 1 : 0, dim) for i = 1:dim]...]))

    template_funcs = [substitute_Polynomial.(interp_funcs_1D, 1, Ref(volumetric_coors[i])) for i = 1:(dim + 1)]

    result_itp_funcs = Polynomial{dim}[]
    for interp_pos in Iterators.product([0:interp_order for i = 1:dim]...)
        last_dim_pos = interp_order - sum(interp_pos)
        last_dim_pos < 0 && continue
        push!(result_itp_funcs, prod([template_funcs[i][interp_pos[i] + 1] for i = 1:dim]) * template_funcs[dim + 1][last_dim_pos + 1])
    end
    return result_itp_funcs
end

# 3  7  8  4 
# 10       12
# 9        11
# 1  5  6  2
#############################
function init_Interpolation_Cube_Serendipity(interp_order::Integer, dim::Integer)
    xs = collect_Basis(dim)
    result_itp_funcs = Polynomial{dim}[]
    # The corner of serendipity is defined ad-hoc each itp order by nature
    if interp_order <= 2
        for coors in Iterators.product([0:1 for i = 1:dim]...)
            this_itp_func = prod((1 .- coors) .- xs)
            for i = 1:(interp_order - 1)
                s = 1 .- 2 .* coors
                this_itp_func *= dot(s, coors) + i / interp_order - sum(s .* xs)
            end
            this_itp_func /= evaluate_Polynomial(this_itp_func, coors) #normalization
            push!(result_itp_funcs, this_itp_func)
        end
    elseif interp_order == 3
        shifted_xs = xs .- 0.5
        for coors in Iterators.product([0:1 for i = 1:dim]...)
            this_itp_func = prod((1 .- coors) .- xs) * (sum(shifted_xs .^ 2) - ((1 / 6) ^ 2 + (dim - 1) * (1 / 2) ^ 2))

            this_itp_func /= evaluate_Polynomial(this_itp_func, coors) #normalization
            push!(result_itp_funcs, this_itp_func)
        end
    else 
        error("Undefined serendipity order")
    end

    for edge_direction = 1:dim
        minor_dims = [i for i = 1:dim if i != edge_direction]
        for minor_coors in Iterators.product([0:1 for i = 1:(dim - 1)]...)
            base_itp_func = prod((1 .- minor_coors) .- xs[minor_dims])
            for itp_pos = 1:(interp_order - 1)
                this_itp_func = prod(Ref(xs[edge_direction]) .- [i / interp_order for i = 0:interp_order if i != itp_pos]) * base_itp_func
                this_coor = [itp_pos / interp_order for i = 1:dim]
                this_coor[minor_dims] .= minor_coors

                this_itp_func /= evaluate_Polynomial(this_itp_func, Tuple(this_coor)) #normalization
                push!(result_itp_funcs, this_itp_func)
            end
        end
    end
    return result_itp_funcs
end

function init_Interpolation_Hermite_1D(interp_order::Integer)
    interp_funcs_1D = Polynomial{1}[]

    T = zeros(CPU_Float, (2 * interp_order), (2 * interp_order))
    rstart, rend = 0, 1
    for this_order = 1:interp_order
        for this_column = this_order:(2 * interp_order)
            T[this_order, this_column] = factorial(this_column - 1)/factorial(this_column - this_order) * rstart ^ (this_column - this_order)
            T[(interp_order + this_order), this_column] = factorial(this_column - 1)/factorial(this_column - this_order) * rend ^ (this_column - this_order)
        end
    end

    T_inv = T^(-1)
    for i = 1:(2 * interp_order)
        this_interp = Polynomial(vec(T_inv[:,i]), [(i,) for i = 0:(2 * interp_order - 1)]) |> check_Clear
        push!(interp_funcs_1D, this_interp)
    end
    return interp_funcs_1D
end
