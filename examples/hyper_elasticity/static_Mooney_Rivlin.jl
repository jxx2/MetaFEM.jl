using MetaFEM
using CUDA
CUDA.allowscalar(false)
initialize_Definitions!()
#------------------------------
# Mesh
#------------------------------
fem_domain = FEM_Domain(dim = 3)
L_box, e_number, LW_ratio = 1., 4, 10
domain_size = (L_box * LW_ratio, L_box, L_box) 
element_number = (Int(e_number * LW_ratio), e_number, e_number)
element_shape = :CUBE

vert, connections = make_Brick(domain_size, element_number, element_shape)
@time ref_mesh = construct_TotalMesh(vert, connections)
#------------------------------
# Define Boundary
#------------------------------
@Takeout (vertices, faces) FROM ref_mesh
facet_IDs = get_BoundaryMesh(ref_mesh)
vIDs = faces.vertex_IDs[:, facet_IDs] 

x1_mean = vec(sum(vertices.x1[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)
x2_mean = vec(sum(vertices.x2[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)
x3_mean = vec(sum(vertices.x3[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)

err_scale = L_box / e_number * 0.01

facet_IDs_left = facet_IDs[(x1_mean .< err_scale) .& (x1_mean .> (.- err_scale))]
facet_IDs_right = facet_IDs[(x1_mean .< (L_box * LW_ratio .+ err_scale)) .& (x1_mean .> (L_box * LW_ratio.- err_scale))]
facet_IDs_front = facet_IDs[(x2_mean .< err_scale) .& (x2_mean .> (.- err_scale))]
facet_IDs_back = facet_IDs[(x2_mean .< (L_box .+ err_scale)) .& (x2_mean .> (L_box .- err_scale))]
facet_IDs_bottom = facet_IDs[(x3_mean .< err_scale) .& (x3_mean .> (.- err_scale))]
facet_IDs_top = facet_IDs[(x3_mean .< (L_box .+ err_scale)) .& (x3_mean .> (L_box .- err_scale))]

wp_ID = add_WorkPiece!(ref_mesh; fem_domain = fem_domain)
fix_bg_ID = add_Boundary!(wp_ID, facet_IDs_left; fem_domain = fem_domain) #left fixed
free_bg_ID = add_Boundary!(wp_ID, vcat(facet_IDs_front, facet_IDs_bottom, facet_IDs_top, facet_IDs_back); fem_domain = fem_domain) #bottom & right & front free
right_bg_ID = add_Boundary!(wp_ID, facet_IDs_right; fem_domain = fem_domain) #top will be loaded
#------------------------------
# Physics
#------------------------------
@Sym d
@External_Sym (dʷ, CONTROLPOINT_VAR) (Pˡ, CONTROLPOINT_VAR) (C10, GLOBAL_VAR) (C01, GLOBAL_VAR) (λ, GLOBAL_VAR) (τᵇ, GLOBAL_VAR)

@Def begin
    F{i,j} = δ{i,j} + d{i;j}
    B{i,j} = F{i,k} * F{j,k} 
    I1 = B{i,i}
    I2 = 0.5 * (B{i,i} ^ 2 - B{i,j} * B{j,i})
    J = F{1,i} * F{2,j} * F{3,k} * ϵ{i,j,k} 
    W = C10 * (I1 - 3 - 2 * log(J)) + C01 * (I2 - 3 - 4 * log(J)) + 0.5 * λ * (J - 1) ^ 2 # Mooney-Rivlin
    P{i,j} = d(W, F{i,j})
end

@Def begin
    WF_domain = - Bilinear(F{i,j}, P{i,j})
    WF_fixed_bdy = τᵇ * Bilinear(d{i}, (dʷ{i} - d{i}))
    WF_right_bdy = Bilinear(d{i}, Pˡ{i,j} * n{j})
end

@time begin
    assign_WorkPiece_WeakForm!(wp_ID, WF_domain; fem_domain = fem_domain)
    assign_Boundary_WeakForm!(wp_ID, fix_bg_ID, WF_fixed_bdy; fem_domain = fem_domain)
    assign_Boundary_WeakForm!(wp_ID, right_bg_ID, WF_right_bdy; fem_domain = fem_domain)
end 
initialize_LocalAssembly!(fem_domain)
#------------------------------
## Assembly
#------------------------------
@time mesh_Classical([wp_ID]; shape = element_shape, itp_type = :Serendipity, itp_order = 2, itg_order = 5, fem_domain = fem_domain)
compile_Updater_GPU(; domain_ID = 1, fem_domain = fem_domain)

@time begin
    for wp in fem_domain.workpieces
        update_Mesh(fem_domain.dim, wp, wp.element_space)
    end
    assemble_Global_Variables!(; fem_domain = fem_domain)
end
#------------------------------
## Run & Gather Data
#------------------------------
fem_domain.linear_solver = x -> iterative_Solve!(x; Sv_func! = bicgstabl_GS!, maxiter = 3000, max_pass = 10, s = 4)

dx = L_box/e_number
err_scale = 0.25

cpts = fem_domain.workpieces[1].mesh.controlpoints
right_cp_IDs = findall((cpts.x1 .> (L_box * LW_ratio) - err_scale * dx) .& (cpts.x1 .< (L_box * LW_ratio) + err_scale * dx))
left_cp_IDs = findall((cpts.x1 .> - err_scale * dx) .& (cpts.x1 .< err_scale * dx))

fem_domain.globalfield.converge_tol = 1e-5

setups = [(1e6, 1e6, 1e8, 30, 4e5), (1e6, 5e6, 1e8, 30, 5e5), (5e6, 1e6, 1e8, 40, 10e5)] #C10, C01, λ, total step, step load
P1s_buffer, d1s_buffer = Vector[], Vector[]
for (_C10, _C01, _λ, total_steps, σ_step) in setups
    P1s = FEM_Float[]
    d1s = FEM_Float[]

    fem_domain.workpieces[1].physics.global_vars[:C10] = _C10
    fem_domain.workpieces[1].physics.global_vars[:C01] = _C01
    fem_domain.workpieces[1].physics.global_vars[:λ] = _λ
    fem_domain.workpieces[1].physics.global_vars[:τᵇ] = 1000 * max(_λ, _C10, _C01) / L_box

    cpts.d1 .= 0 
    cpts.d2 .= 0
    cpts.d3 .= 0
    assemble_X!(fem_domain.workpieces, fem_domain.globalfield)
    for i = 1:total_steps
        println("Global step $i")

        σ_load = σ_step * i
        cpts.Pˡ1 .= σ_load

        update_OneStep!(fem_domain.time_discretization; fem_domain = fem_domain, max_iter = 7)
        dessemble_X!(fem_domain.workpieces, fem_domain.globalfield)

        step_d1 = sum(cpts.d1[right_cp_IDs]) / (L_box * LW_ratio * length(right_cp_IDs))
        push!(d1s, step_d1)
        push!(P1s, σ_load)
    end
    push!(P1s_buffer, P1s)
    push!(d1s_buffer, d1s)
end
#------------------------------
## Plots
#------------------------------
Jac(l1, _C10, _C01, _λ) = l1 * (-(2 * _C10 + 2 * _C01 * l1 ^ 2 + _λ * l1) + sqrt((2 * _C10 + 2 * _C01 * l1 ^ 2 + _λ * l1) ^ 2 + 8 * _C01 * (_λ + 2 * _C10 + 4 * _C01))) / (4 * _C01)
mooney_Rivlin(l1, _C10, _C01, _λ)  = 2 * _C10 * l1 + 4 * _C01 * Jac(l1, _C10, _C01, _λ) + (_λ * (Jac(l1, _C10, _C01, _λ) - 1) - 2 * _C10 - 4 * _C01) / l1

using Plots
fig = plot(; size = (900,800), title = "Uniaxial loading on Mooney-Rivlin hyper-elastic materials.", xlims = (0., 2.2), xticks = 0:0.2:2, ylims = (0., 35), xlabel = "Elongation", ylabel = "Nominal stress(MPa)")
case_num = length(P1s_buffer)
for i = 1:case_num
    color_val = (i / case_num + 1) / 2
    P1s, d1s = P1s_buffer[i], d1s_buffer[i]
    (_C10, _C01, _λ, _, _) = setups[i]
    scatter!(fig, d1s, P1s ./ 1e6, markershape = :circle, markersize = 6, color = RGBA(color_val, 0, 0, 1), label = "numerical, C10 = $_C10 Pa, C01 = $_C01 Pa, λ = $_λ Pa")
    plot!(fig, d1s, mooney_Rivlin.(d1s .+ 1, _C10, _C01, _λ) ./ 1e6, color = RGBA(0, 0.5, color_val, 1), label = "analytical, C10 = $_C10 Pa, C01 = $_C01 Pa, λ = $_λ Pa")
end
fig.subplots[1].attr[:legend_position] = (0.15, 0.9)
fig
#------------------------------
## Save outputs
#------------------------------
png(fig, joinpath(@__DIR__, "Mooney-Rivlin_Tensile_Test.png"))
