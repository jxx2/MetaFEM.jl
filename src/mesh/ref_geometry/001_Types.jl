mutable struct Geo_Vertex2D
    x1::FEM_Float
    x2::FEM_Float
end

mutable struct Geo_Vertex3D
    x1::FEM_Float
    x2::FEM_Float
    x3::FEM_Float
end

mutable struct Geo_Segment
    vertex_IDs::Vector{FEM_Int}
end

mutable struct Geo_Face
    vertex_IDs::Vector{FEM_Int}
    segment_IDs::Vector{FEM_Int} #has an order
end

mutable struct Geo_Block
    vertex_IDs::Vector{FEM_Int}
    segment_IDs::Vector{FEM_Int} #has an order
    face_IDs::Vector{FEM_Int}
end

mutable struct Geo_TotalMesh2D{ArrayType} <: FEM_Geometry{ArrayType}
    vertices::FEM_Table{ArrayType}
    segments::FEM_Table{ArrayType}
    faces::FEM_Table{ArrayType}
end

mutable struct Geo_TotalMesh3D{ArrayType} <: FEM_Geometry{ArrayType}
    vertices::FEM_Table{ArrayType}
    segments::FEM_Table{ArrayType}
    faces::FEM_Table{ArrayType}
    blocks::FEM_Table{ArrayType}
end

mutable struct Geo_BoundaryMesh2D{ArrayType} <: FEM_Geometry{ArrayType}
    vertices::FEM_Table{ArrayType}
    segments::FEM_Table{ArrayType}
end

mutable struct Geo_BoundaryMesh3D{ArrayType} <: FEM_Geometry{ArrayType}
    vertices::FEM_Table{ArrayType}
    segments::FEM_Table{ArrayType}
    faces::FEM_Table{ArrayType}
end

function Geo_TotalMesh2D(::Type{ArrayType}, mesh_type::Symbol) where {ArrayType}
    if mesh_type == :SIMPLEX
        vertex_per_face = 3
    elseif mesh_type == :CUBE
        vertex_per_face = 4
    else
        error("Undefined mesh type")
    end
    vertex_example = Geo_Vertex2D(ntuple(x -> FEM_Float(0.), 2)...)
    segment_example = Geo_Segment(zeros(FEM_Int, 2))
    face_example = Geo_Face(zeros(FEM_Int, vertex_per_face), zeros(FEM_Int, vertex_per_face)) 
    examples = (vertex_example, segment_example, face_example)
    return Geo_TotalMesh2D((construct_FEM_Table.(ArrayType, examples))...)
end

function Geo_TotalMesh3D(::Type{ArrayType}, mesh_type::Symbol) where {ArrayType}
    if mesh_type == :SIMPLEX
        vertex_per_face = 3
        vertex_per_block = 4
        segment_per_block = 6
        face_per_block = 4
    elseif mesh_type == :CUBE
        vertex_per_face = 4
        vertex_per_block = 8
        segment_per_block = 12
        face_per_block = 6
    else
        error("Undefined mesh type")
    end
    vertex_example = Geo_Vertex3D(ntuple(x -> FEM_Float(0.), 3)...)
    segment_example = Geo_Segment(zeros(FEM_Int, 2))
    face_example = Geo_Face(zeros(FEM_Int, vertex_per_face), zeros(FEM_Int, vertex_per_face))
    block_example = Geo_Block(zeros(FEM_Int, vertex_per_block), zeros(FEM_Int, segment_per_block), zeros(FEM_Int, face_per_block))
    examples = (vertex_example, segment_example, face_example, block_example)
    return Geo_TotalMesh3D((construct_FEM_Table.(ArrayType, examples))...)
end

function Geo_BoundaryMesh2D(::Type{ArrayType}) where {ArrayType}
    vertex_example = Geo_Vertex2D(ntuple(x -> FEM_Float(0.), 2)...)
    segment_example = Geo_Segment(zeros(FEM_Int, 2))
    examples = (vertex_example, segment_example)
    return Geo_BoundaryMesh2D((construct_FEM_Table.(ArrayType, examples))...)
end

function Geo_BoundaryMesh3D(::Type{ArrayType}) where {ArrayType}
    vertex_example = Geo_Vertex3D(ntuple(x -> FEM_Float(0.), 3)...)
    segment_example = Geo_Segment(zeros(FEM_Int, 2))
    face_example = Geo_Face(zeros(FEM_Int, 3), zeros(FEM_Int, 3))
    examples = (vertex_example, segment_example, face_example)
    return Geo_BoundaryMesh3D((construct_FEM_Table.(ArrayType, examples))...)
end