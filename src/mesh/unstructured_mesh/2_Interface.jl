"""
    mesh_Classical(wp_IDs; shape::Symbol, itp_type::Symbol = :Lagrange, itp_order::Integer, itg_order::Integer, fem_domain::FEM_Domain)

The function generates the mesh of order `itp_order`, interpolation type `itp_type` = `:Lagrange`/`:Serendipity` with gaussian quadrature of order `itg_order` on `fem_domain`.`workpieces`[`wp_IDs`].
The dimension and mesh type (`:CUBE`/`:SIMPLEX`) will follow the first order mesh of each `WorkPiece`.
"""
function mesh_Classical(wp_IDs; shape::Symbol, itp_type::Symbol = :Lagrange, itp_order::Integer, itg_order::Integer, fem_domain::FEM_Domain)
    dim = fem_domain.dim
    for wp in fem_domain.workpieces[wp_IDs]
        this_space = wp.element_space = initialize_Classical_Element(dim, shape, itp_order, wp.max_sd_order, itg_order; itp_type = itp_type)

        controlpoints = declare_Basic_ControlPoint(dim, wp)
        facets = declare_Basic_Facet(dim, this_space)
        elements = declare_Basic_Element(dim, this_space)
        cp_cp_2_sparseID = construct_GPUDict([:sparseID => FEM_Int(0)])

        bg_fIDs = Dict{FEM_Int, CuVector{FEM_Int}}()
        variable_size = FEM_Int(0)
        wp.mesh = @Construct Basic_WP_Mesh

        if dim == 2
            allocate_Basic_WP_Mesh_2D(wp, this_space)

            evaled_max_sd_order = length(BASE_KERNELS_2D)
            if wp.max_sd_order > evaled_max_sd_order
                append!(BASE_KERNELS_2D, [Core.eval(@__MODULE__, gen_Kernel_Itpval(i, 2)) for i = evaled_max_sd_order:(wp.max_sd_order)])
            end
        elseif dim == 3
            allocate_Basic_WP_Mesh_3D(wp, this_space)

            evaled_max_sd_order = length(BASE_KERNELS_3D)
            if wp.max_sd_order > evaled_max_sd_order
                append!(BASE_KERNELS_3D, [Core.eval(@__MODULE__, gen_Kernel_Itpval(i, 3)) for i = evaled_max_sd_order:(wp.max_sd_order)])
            end
        else
            error("Undefined dimension")
        end
        println("Allocate ", volumeof(controlpoints), " bytes for physical DOF and ", (facets, elements, cp_cp_2_sparseID) .|> volumeof |> sum, " bytes for geometry")
    end
end

function declare_Basic_ControlPoint(dim::Integer, wp::WorkPiece)
    @Takeout (local_innervar_infos, local_extervars) FROM wp.local_assembly

    global_cpID, element_num = (0, 0) .|> FEM_Int
    if dim == 2
        x1, x2 = (0., 0.) .|> FEM_Float
        controlpoints = @Construct Basic_ControlPoint2D
    elseif dim == 3
        x1, x2, x3 = (0., 0., 0.) .|> FEM_Float
        controlpoints = @Construct Basic_ControlPoint3D
    end
    local_innervars = getindex.(local_innervar_infos, 1)
    return construct_GPUTable(controlpoints, Symbol[local_innervars..., local_extervars...] .=> FEM_Float(0.))
end

function declare_Basic_Facet(dim::Integer, this_space::Classical_Discretization)
    @Takeout (bdy_itg_func_num, bdy_ref_itp_vals) FROM this_space
    tangent_directions = zeros(FEM_Float, bdy_itg_func_num, dim, dim - 1)
    normal_directions = zeros(FEM_Float, bdy_itg_func_num, dim)

    element_ID, element_eindex = (0, 0) .|> FEM_Int
    integral_vals = zeros(FEM_Float, size(this_space.bdy_ref_itp_vals[1]))
    jacobian = zeros(FEM_Float, dim, dim, bdy_itg_func_num)
    inverse_jacobian = zeros(FEM_Float, dim, dim, bdy_itg_func_num)
    el_dets = zeros(FEM_Float, bdy_itg_func_num)
    bdy_dets = zeros(FEM_Float, bdy_itg_func_num)
    integral_weights = zeros(FEM_Float, bdy_itg_func_num)

    outer_element_ID, outer_element_eindex = (0, 0) .|> FEM_Int

    boundaries = @Construct Basic_Facet
    return construct_GPUTable(boundaries, Symbol[] .=> Array[])
end

function declare_Basic_Element(dim::Integer, this_space::Classical_Discretization)
    @Takeout (itp_func_num, itg_func_num, ref_itp_vals) FROM this_space

    controlpoint_IDs, global_cpIDs = zeros(FEM_Int, itp_func_num), zeros(FEM_Int, itp_func_num)
    sparse_IDs_by_el = zeros(FEM_Int, itp_func_num, itp_func_num)

    integral_vals = zeros(FEM_Float, size(ref_itp_vals))
    integral_weights  = zeros(FEM_Float, itg_func_num)

    # facet_IDs = zeros(FEM_Int, length(this_space.bdy_ref_itp_vals))
    jacobian = zeros(FEM_Float, dim, dim, itg_func_num)
    inverse_jacobian = zeros(FEM_Float, dim, dim, itg_func_num)
    dets = zeros(FEM_Float, itg_func_num)

    elements = @Construct Basic_Element
    return construct_GPUTable(elements, Symbol[] .=> Array[])
end

"""
    update_Mesh(dim::Integer, wp::WorkPiece, this_space::Classical_Discretization)

This function updates Jacobians and interpolation values.
"""
function update_Mesh(dim::Integer, wp::WorkPiece, this_space::Classical_Discretization)
    if dim == 2
        itpval_kernel = BASE_KERNELS_2D[wp.max_sd_order]        
        update_BasicElements_2D(wp.mesh, this_space, itpval_kernel)
        update_BasicBoundary_2D(wp.mesh, this_space, itpval_kernel)
    elseif dim == 3
        itpval_kernel = BASE_KERNELS_3D[wp.max_sd_order]
        update_BasicElements_3D(wp.mesh, this_space, itpval_kernel)
        update_BasicBoundary_3D(wp.mesh, this_space, itpval_kernel)
    end
end

