mutable struct GeneralAlpha <: FEM_Temporal_Discretization
    alpha_params::Tuple{Vararg{FEM_Float}}
    gamma_params::Tuple{Vararg{FEM_Float}}
    beta_params::Vector{FEM_Float}
    K_params::Vector{FEM_Float}
end
GeneralAlpha(; dissipative::Bool = false) = GeneralAlpha((1., 1., 1.) .|> FEM_Float, (dissipative ? (1., 1.) : (0.5, 0.5)) .|> FEM_Float, FEM_Float[], FEM_Float[])
# GeneralAlpha() = GeneralAlpha((1., 1., 1.) .|> FEM_Float, (0.5, 0.5) .|> FEM_Float, FEM_Float[], FEM_Float[])

function update_Time!(globalfield::GlobalField, time_discretization::GeneralAlpha)
    globalfield.t += globalfield.dt

    prod_gamma = [prod(time_discretization.gamma_params[1:i]) for i = 0:globalfield.max_time_level]
    dt_params = [globalfield.dt ^ i for i = 0:globalfield.max_time_level]
    time_discretization.beta_params = 1 ./ (prod_gamma .* dt_params)

    time_discretization.K_params = time_discretization.alpha_params[1:(globalfield.max_time_level + 1)] .* time_discretization.beta_params
end

function initialize_dx!(globalfield::GlobalField, time_discretization::GeneralAlpha)
    @Takeout (max_time_level, basicfield_size, dt, x, dx) FROM globalfield
    @Takeout (gamma_params) FROM time_discretization
    dx.= 0.
    for t_level = max_time_level:-1:1
        low_start, low_final   = (t_level - 1) * basicfield_size + 1,  t_level      * basicfield_size
        high_start, high_final =  t_level      * basicfield_size + 1, (t_level + 1) * basicfield_size

        dx[low_start:low_final] .= dt * (x[high_start:high_final] .+ gamma_params[t_level] * dx[high_start:high_final])
    end
end

function update_dx!(globalfield::GlobalField, delta_x::CuVector, time_discretization::GeneralAlpha)
    @Takeout (max_time_level, basicfield_size, x, dx) FROM globalfield
    @Takeout beta_params FROM time_discretization
    for t_level = 0:max_time_level
        start_id, final_id = t_level * basicfield_size + 1, (t_level + 1) * basicfield_size
        dx[start_id:final_id] .+= beta_params[t_level + 1] * delta_x
    end
end

function update_x_star!(globalfield::GlobalField, time_discretization::GeneralAlpha)
    @Takeout (max_time_level, basicfield_size, dt, x, dx) FROM globalfield
    @Takeout (alpha_params) FROM time_discretization
    globalfield.x_star .= x
    for t_level = 0:max_time_level
        start_id, final_id = t_level * basicfield_size + 1, (t_level + 1) * basicfield_size
        globalfield.x_star[start_id:final_id] .+= alpha_params[t_level + 1] * dx[start_id:final_id]
    end
end

normalized_norm(x) = norm(x) / sqrt(length(x))

"""
    update_OneStep!(time_discretization::GeneralAlpha; max_iter::Integer = 4, fem_domain::FEM_Domain)

This function calculates `K(Δx)=d` and updates `x += (Δx)`. `max_iter` determines the maximum iteration for Δx with different `K` and `d` in nonlinear cases.
The converge tolerance is determined by `fem_domain`.`globalfield`.`converge_tol` while the linear solver is `fem_domain`.`linear_solver`.
"""
function update_OneStep!(time_discretization::GeneralAlpha; max_iter::Integer = 4, fem_domain::FEM_Domain)
    @Takeout (workpieces, globalfield) FROM fem_domain

    update_Time!(globalfield, time_discretization)
    initialize_dx!(globalfield, time_discretization)
    fem_domain.K_linear_func(time_discretization; fem_domain = fem_domain)
    counter = -1
    while true
        update_x_star!(globalfield, time_discretization)
        @time fem_domain.K_nonlinear_func(time_discretization; fem_domain = fem_domain)
        res = normalized_norm(globalfield.residue)

        println(repeat("_", 100))
        println("step $(counter += 1) residue = $res")

        (res < globalfield.converge_tol || counter > max_iter) && break

        @time delta_x = fem_domain.linear_solver(globalfield)
        update_dx!(globalfield, .- delta_x, time_discretization)
    end
    globalfield.x .+= globalfield.dx
end