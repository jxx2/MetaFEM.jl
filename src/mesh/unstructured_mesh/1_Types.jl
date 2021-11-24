mutable struct Basic_ControlPoint2D 
    x1::FEM_Float
    x2::FEM_Float

    global_cpID::FEM_Int #cp_IDs of FIRST local variable, the rest variables need to be shifted
end

mutable struct Basic_ControlPoint3D 
    x1::FEM_Float
    x2::FEM_Float
    x3::FEM_Float

    global_cpID::FEM_Int #cp_IDs of FIRST local variable, the rest variables need to be shifted
end

mutable struct Basic_Facet
    tangent_directions::Array{FEM_Float, 3} #normal_direction
    normal_directions::Array{FEM_Float, 2}

    jacobian::Array{FEM_Float, 3}
    inverse_jacobian::Array{FEM_Float, 3}
    el_dets::Vector{FEM_Float}
    bdy_dets::Vector{FEM_Float}

    element_ID::FEM_Int
    element_eindex::FEM_Int
    integral_vals::Array{FEM_Float} #diff vals separate
    integral_weights::Vector{FEM_Float}

    # outer_jacobian::Array{FEM_Float, 3}
    # outer_inverse_jacobian::Array{FEM_Float, 3}
    # outer_el_dets::Vector{FEM_Float}
    # outer_bdy_dets::Vector{FEM_Float}   
    outer_element_ID::FEM_Int #may not applicable, but reserved for Discontinuous Galerkin
    outer_element_eindex::FEM_Int
    # outer_integral_vals::Array{FEM_Float}
end

mutable struct Basic_Element
    #common part about element
    controlpoint_IDs::Vector{FEM_Int}

    global_cpIDs::Vector{FEM_Int}
    sparse_IDs_by_el::Array{FEM_Int, 2}

    integral_vals::Array{FEM_Float} #diff vals separate
    integral_weights::Vector{FEM_Float}
    #for basic mesh
    # facet_IDs::Vector{FEM_Int}
    jacobian::Array{FEM_Float, 3}
    inverse_jacobian::Array{FEM_Float, 3}
    dets::Vector{FEM_Float}
end

mutable struct Basic_WP_Mesh <: FEM_WP_Mesh
    controlpoints::GPUTable
    facets::GPUTable #boundaries 
    elements::GPUTable
    # cp_pos_2_el_pos::GPUDict
    # cp_cp_2_sparseID::GPUDict

    bg_fIDs::Dict{FEM_Int, CuVector{FEM_Int}}
    variable_size::FEM_Int
end
