struct Classical_Element_Structure
    vertex_cp_ids::Array{FEM_Int} 
    segment_cp_ids::Array{FEM_Int} 
    face_cp_ids::Array{FEM_Int} 
    block_cp_ids::Array{FEM_Int} 
    
    segment_cp_pos::Array{FEM_Float} 
    face_cp_pos::Array{FEM_Float} 
    block_cp_pos::Array{FEM_Float} 

    segment_start_vertex::Array{FEM_Int}
    face_start_segments::Array{FEM_Int}
end

mutable struct Classical_Discretization{ArrayType} <: FEM_Spatial_Discretization{ArrayType}
    element_attributes::Dict{Symbol, Any} 
    #topology 
    element_structure::Classical_Element_Structure
    #interpolation
    itp_func_num::FEM_Int
    itp_funcs::Vector{Polynomial}
    #domain integral
    itg_func_num::FEM_Int
    itg_weight::ArrayType

    ref_itp_vals::ArrayType
    #boundary integral
    bdy_itg_func_num::FEM_Int
    bdy_itg_weights::Vector{ArrayType}
    bdy_tangent_directions::Vector{ArrayType} #itg_ID, local vector, tangent ID (2D only 1, 3D 2)...

    bdy_ref_itp_vals::Vector{ArrayType} #itg_ID, bdy_cp_ID, diff mode: 1, partialX, partial Y, ...
end

initialize_Classical_Element(dim::Integer, shape::Symbol, itp_order::Integer, max_sd_order::Integer, itg_order::Integer; itp_type::Symbol = :Lagrange) = 
initialize_Classical_Element(DEFAULT_ARRAYINFO._type, dim, shape, itp_order, max_sd_order, itg_order; itp_type = :itp_type)
function initialize_Classical_Element(::Type{ArrayType}, dim::Integer, shape::Symbol, itp_order::Integer, max_sd_order::Integer, itg_order::Integer; itp_type::Symbol = :Lagrange) where {ArrayType}
    if shape == :CUBE
        if itp_type == :Lagrange
            if dim == 2
                element_structure = init_Structure_Cube2D_Lagrange(itp_order)
            elseif dim == 3
                element_structure = init_Structure_Cube3D_Lagrange(itp_order)
            else
                error("Wrong dimension for structure")
            end
            itp_funcs = init_Interpolation_Cube_Lagrange(itp_order, dim)
        elseif itp_type == :Serendipity
            if dim == 2
                element_structure = init_Structure_Cube2D_Serendipity(itp_order)
            elseif dim == 3
                element_structure = init_Structure_Cube3D_Serendipity(itp_order)
            else
                error("Wrong dimension for structure")
            end
            itp_funcs = init_Interpolation_Cube_Serendipity(itp_order, dim)
        end
        itg_pos, itg_weight = init_Domain_Integration_Cube_Gauss(itg_order, dim)
        bdy_itg_pos, bdy_itg_weights, bdy_tangent_directions = init_Boundary_Integration_Cube_Gauss(itg_order, dim)
    elseif shape == :SIMPLEX
        if dim == 2
            element_structure = init_Structure_Triangle_Lagrange(itp_order)
            itg_pos, itg_weight = init_Domain_Integration_Triangle_Gauss(itg_order)
            bdy_itg_pos, bdy_itg_weights, bdy_tangent_directions = init_Boundary_Integration_Triangle_Gauss(itg_order)
        elseif dim == 3
            element_structure = init_Structure_Tetrahedron_Lagrange(itp_order)
            itg_pos, itg_weight = init_Domain_Integration_Tetrahedron_Gauss(itg_order)
            bdy_itg_pos, bdy_itg_weights, bdy_tangent_directions = init_Boundary_Integration_Tetrahedron_Gauss(itg_order)
        else
            error("Wrong dimension for structure")
        end
        itp_funcs = init_Interpolation_Simplex_Lagrange(itp_order, dim)
    end
    element_attributes = Dict(:dim => dim, :shape => shape, :itp_type => itp_type, :itp_order => itp_order)
    itp_func_num, itg_func_num, bdy_itg_func_num = (itp_funcs, itg_weight, bdy_itg_weights[1]) .|> length
    itg_weight, bdy_itg_weights, bdy_tangent_directions = (FEM_convert(ArrayType, itg_weight), FEM_convert.(ArrayType, bdy_itg_weights), FEM_convert.(ArrayType, bdy_tangent_directions))

    ref_itp_vals = FEM_convert(ArrayType, evaluate_Itp_Funcs(itp_funcs, max_sd_order, itg_pos))
    bdy_ref_itp_vals = FEM_convert.(ArrayType, evaluate_Itp_Funcs.(Ref(itp_funcs), max_sd_order, bdy_itg_pos))
    return @Construct Classical_Discretization{ArrayType}
end

function evaluate_Itp_Funcs(itp_funcs::Vector{Polynomial{dim}}, max_sd_order::Integer, itg_pos::Vector) where dim
    itp_func_num = length(itp_funcs)
    itg_func_num = size(itg_pos, 1)

    grad_size = fill(max_sd_order + 1, dim)
    itp_gradient_funcs = fill(Polynomial{dim}[], grad_size...) #Note this one is allowed because the whole array pointer will be replaced right later
    ref_itp_vals = zeros(FEM_Float, itg_func_num, itp_func_num, grad_size...)

    for diff_orders in Iterators.product([0:max_sd_order for i = 1:dim]...)
        diff_ids = diff_orders .+ 1
        itp_gradient_func_batch = derivative.(itp_funcs, Ref(diff_orders))
        itp_gradient_funcs[diff_ids...] = itp_gradient_func_batch
        ref_itp_vals[:, :, diff_ids...] .= [evaluate_Polynomial(this_itp_func, pos) for pos in itg_pos, this_itp_func in itp_gradient_func_batch]
    end
    return ref_itp_vals
end
