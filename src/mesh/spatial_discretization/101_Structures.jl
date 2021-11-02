function init_Structure_Cube2D_Lagrange(itp_order::Integer)
    cp_per_dim = itp_order + 1

    cp_per_vertex = 1
    vertex_cp_ids = [1 cp_per_dim cp_per_dim ^ 2 cp_per_dim * (cp_per_dim - 1) + 1]

    cp_per_segment = cp_per_dim - 2
    segment_cp_ids = hcat([i + 1 for i = 1:cp_per_segment], 
    [cp_per_dim * (i + 1) for i = 1:cp_per_segment],
    [cp_per_dim * (cp_per_dim - 1) + i + 1 for i = 1:cp_per_segment],
    [cp_per_dim * i + 1 for i = 1:cp_per_segment])
    segment_cp_pos = hcat([[1. - i / itp_order, i / itp_order] for i = 1:cp_per_segment]...)
    segment_start_vertex = [1, 2, 4, 1]

    cp_per_face = cp_per_segment ^ 2
    face_cp_ids = [j * cp_per_dim + i + 1 for j = 1:cp_per_segment for i = 1:cp_per_segment] # i -> x j -> y
    face_cp_pos = hcat([[
                          (1. - i / itp_order) * (1. - j / itp_order), 
                          (     i / itp_order) * (1. - j / itp_order),
                          (     i / itp_order) * (     j / itp_order), 
                          (1. - i / itp_order) * (     j / itp_order)
                        ] for j = 1:cp_per_segment for i = 1:cp_per_segment]...)
    face_start_segments = zeros(FEM_Int, 0, 1)

    cp_per_block = 0
    block_cp_ids = zeros(FEM_Int, cp_per_block, 0)
    block_cp_pos = zeros(FEM_Float, 0, cp_per_block)

    return @Construct Classical_Element_Structure
end

function init_Structure_Cube3D_Lagrange(itp_order::Integer)
    cp_per_dim = itp_order + 1

    cp_per_vertex = 1
    v_cp_ids_1D = [1 cp_per_dim cp_per_dim ^ 2 cp_per_dim * (cp_per_dim - 1) + 1]
    vertex_cp_ids = hcat(v_cp_ids_1D, v_cp_ids_1D .+ (cp_per_dim - 1) * cp_per_dim ^ 2)

    cp_per_segment = cp_per_dim - 2
    segment_cp_ids = hcat([i + 1 for i = 1:cp_per_segment],
    [cp_per_dim * (i + 1) for i = 1:cp_per_segment],
    [cp_per_dim * (cp_per_dim - 1) + i + 1 for i = 1:cp_per_segment],
    [cp_per_dim * i + 1 for i = 1:cp_per_segment],

    [cp_per_dim ^ 2 * i + 1 for i = 1:cp_per_segment],
    [cp_per_dim ^ 2 * i + cp_per_dim for i = 1:cp_per_segment],
    [cp_per_dim ^ 2 * i + cp_per_dim ^ 2 for i = 1:cp_per_segment],
    [cp_per_dim ^ 2 * i + cp_per_dim * (cp_per_dim - 1) + 1 for i = 1:cp_per_segment],

    [(cp_per_dim - 1) * cp_per_dim ^ 2 + i + 1 for i = 1:cp_per_segment],
    [(cp_per_dim - 1) * cp_per_dim ^ 2 + cp_per_dim * (i + 1) for i = 1:cp_per_segment],
    [(cp_per_dim - 1) * cp_per_dim ^ 2 + cp_per_dim * (cp_per_dim - 1) + i + 1 for i = 1:cp_per_segment],
    [(cp_per_dim - 1) * cp_per_dim ^ 2 + cp_per_dim * i + 1 for i = 1:cp_per_segment]
    )

    segment_cp_pos = hcat([[1. - i / itp_order, i / itp_order] for i = 1:cp_per_segment]...)
    segment_start_vertex = [1, 2, 4, 1,
    1, 2, 3, 4,
    5, 6, 8, 5]

    cp_per_face = cp_per_segment ^ 2
    face_cp_ids = hcat([j * cp_per_dim + i + 1 for j = 1:cp_per_segment for i = 1:cp_per_segment],
    [j * cp_per_dim ^ 2 + i + 1 for j = 1:cp_per_segment for i = 1:cp_per_segment],
    [j * cp_per_dim ^ 2 + i * cp_per_dim + cp_per_dim for j = 1:cp_per_segment for i = 1:cp_per_segment],
    [j * cp_per_dim ^ 2 + i + 1 + (cp_per_dim - 1) * cp_per_dim for j = 1:cp_per_segment for i = 1:cp_per_segment],
    [j * cp_per_dim ^ 2 + i * cp_per_dim + 1 for j = 1:cp_per_segment for i = 1:cp_per_segment],
    [j * cp_per_dim + i + 1 + (cp_per_dim - 1) * cp_per_dim ^ 2 for j = 1:cp_per_segment for i = 1:cp_per_segment]
    )
    face_cp_pos = hcat([[
                          (1. - i / itp_order) * (1. - j / itp_order), 
                          (     i / itp_order) * (1. - j / itp_order),
                          (     i / itp_order) * (     j / itp_order), 
                          (1. - i / itp_order) * (     j / itp_order)
                        ] for j = 1:cp_per_segment for i = 1:cp_per_segment]...)

    face_start_segments = hcat([4, 1], [5, 1], [6, 2], [8, 3], [5, 4], [12, 9]) 

    cp_per_block = cp_per_segment ^ 3

    block_cp_ids = [k * cp_per_dim ^ 2 + j * cp_per_dim + i + 1 for k = 1:cp_per_segment for j = 1:cp_per_segment for i = 1:cp_per_segment]
    block_cp_pos = hcat([[
                          (1. - i / itp_order) * (1. - j / itp_order) * (1. - k / itp_order),
                          (     i / itp_order) * (1. - j / itp_order) * (1. - k / itp_order),
                          (     i / itp_order) * (     j / itp_order) * (1. - k / itp_order),
                          (1. - i / itp_order) * (     j / itp_order) * (1. - k / itp_order),  
                          (1. - i / itp_order) * (1. - j / itp_order) * (     k / itp_order),
                          (     i / itp_order) * (1. - j / itp_order) * (     k / itp_order),
                          (     i / itp_order) * (     j / itp_order) * (     k / itp_order),
                          (1. - i / itp_order) * (     j / itp_order) * (     k / itp_order)        
                        ] for k = 1:cp_per_segment for j = 1:cp_per_segment for i = 1:cp_per_segment]...)
    return @Construct Classical_Element_Structure
end

function init_Structure_Triangle_Lagrange(itp_order::Integer)
    cp_per_dim = itp_order + 1

    cp_per_vertex = 1
    vertex_cp_ids = [1 cp_per_dim FEM_Int(cp_per_dim * (cp_per_dim + 1) / 2)]

    cp_per_segment = (cp_per_dim - 2)
    segment_cp_ids = hcat([i + 1 for i = 1:cp_per_segment],
    [FEM_Int((2 * cp_per_dim - i) * (i + 1) / 2) for i = 1:cp_per_segment],
    [FEM_Int((2 * cp_per_dim - i + 1) * i / 2 + 1) for i = 1:cp_per_segment])

    segment_cp_pos = hcat([[1. - i / itp_order, i / itp_order] for i = 1:cp_per_segment]...)
    segment_start_vertex = [1, 2, 1]

    cp_per_face = FEM_Int((cp_per_segment - 1) * cp_per_segment / 2)
    this_id = 0
    face_cp_ids = zeros(FEM_Int, cp_per_face)
    face_cp_pos = zeros(FEM_Float, 3, cp_per_face)
    for j = 1:(cp_per_segment - 1)
        for i = 1:(cp_per_segment - j)
            this_id += 1
            k = itp_order - j - i 
            face_cp_ids[this_id] = (2 * cp_per_dim - i + 1) * i / 2 + i + 1
            face_cp_pos[:, this_id] .= (k, i, j) ./ itp_order
        end
    end
    face_start_segments = zeros(FEM_Int, 0, 1)

    cp_per_block = 0
    block_cp_ids = zeros(FEM_Int, cp_per_block, 0)
    block_cp_pos = zeros(FEM_Float, 0, cp_per_block)

    return @Construct Classical_Element_Structure
end

function init_Structure_Tetrahedron_Lagrange(itp_order::Integer)
    cp_per_dim = itp_order + 1

    cp_per_vertex = 1
    vertex_cp_ids = [1 cp_per_dim FEM_Int(cp_per_dim * (cp_per_dim + 1) / 2) FEM_Int(cp_per_dim * (cp_per_dim + 1) * (cp_per_dim + 2) / 6)]

    cp_per_segment = (cp_per_dim - 2)
    segment_cp_ids = zeros(FEM_Int, cp_per_segment, 6)
    segment_cp_ids[:, 1] .= [i + 1 for i = 1:cp_per_segment]
    segment_cp_ids[:, 2] .= [FEM_Int((2 * cp_per_dim - i) * (i + 1) / 2) for i = 1:cp_per_segment]
    segment_cp_ids[:, 3] .= [FEM_Int((2 * cp_per_dim - i + 1) * i / 2 + 1) for i = 1:cp_per_segment] #4,5,6 allocate incrementally

    segment_cp_pos = hcat([[1. - i / itp_order, i / itp_order] for i = 1:cp_per_segment]...)
    segment_start_vertex = [1, 2, 1, 1, 2, 3]

    cp_per_face = FEM_Int((cp_per_segment - 1) * cp_per_segment / 2)
    cp_per_block = FEM_Int((cp_per_segment - 2) * (cp_per_segment - 1) * cp_per_segment / 6)

    this_id = 0
    face_cp_ids = zeros(FEM_Int, cp_per_face, 4)
    face_cp_pos = zeros(FEM_Float, 3, cp_per_face)
    block_cp_ids = zeros(FEM_Int, cp_per_block)
    block_cp_pos = zeros(FEM_Float, 4, cp_per_block)
   
    for j = 1:cp_per_segment
        for i = 1:cp_per_segment
            k = itp_order - j - i 
            k <= 0 && continue

            this_id += 1
            face_cp_ids[this_id, 1] = (2 * cp_per_dim - i + 1) * i / 2 + i + 1
            face_cp_pos[:, this_id] .= (k, i, j) ./ itp_order
        end
    end

    last_final_cp_id = FEM_Int(cp_per_dim * (cp_per_dim + 1) / 2)
    face_last_id = 0
    block_last_id = 0
    for k = 1:cp_per_segment
        segment_cp_ids[k, 4] = last_final_cp_id + 1
        current_pos_in_layer = cp_per_dim - k
        segment_cp_ids[k, 5] = last_final_cp_id + current_pos_in_layer

        face_cp_ids[(face_last_id + 1):(face_last_id + (cp_per_segment - k)), 2] .= (segment_cp_ids[k, 4] + 1):(segment_cp_ids[k, 5] - 1)

        for j = 1:(cp_per_segment - k)
            face_cp_ids[face_last_id + j, 4] = last_final_cp_id + current_pos_in_layer + 1
            current_pos_in_layer += cp_per_dim - k - j
            face_cp_ids[face_last_id + j, 3] = last_final_cp_id + current_pos_in_layer

            (cp_per_segment - k - j) <= 0 && continue
            block_range = (block_last_id + 1):(block_last_id + (cp_per_segment - k - j))
            block_cp_ids[block_range] .= (face_cp_ids[face_last_id + j, 4] + 1):(face_cp_ids[face_last_id + j, 3] - 1)
            block_cp_pos[1, block_range] .= (cp_per_segment - k - j):1
            block_cp_pos[2, block_range] .= 1:(cp_per_segment - k - j)
            block_cp_pos[3, block_range] .= j
            block_cp_pos[4, block_range] .= k

            block_last_id += (cp_per_segment - k - j)
        end
        last_final_cp_id += current_pos_in_layer + 1
        segment_cp_ids[k, 6] = last_final_cp_id 
    end
    block_cp_pos ./= itp_order
    face_start_segments = hcat([3, 1], [4, 1], [5, 2], [4, 3]) 
    
    return @Construct Classical_Element_Structure
end

function init_Structure_Cube2D_Serendipity(itp_order::Integer)
    cp_per_dim = itp_order + 1

    cp_per_vertex = 1
    vertex_cp_ids = [1 2 4 3]

    cp_per_segment = cp_per_dim - 2
    segment_orders = [0, 3, 1, 2]
    cp_per_segment = cp_per_dim - 2
    segment_cp_ids = [segment_orders[j] * cp_per_segment + 4 + i for i = 1:cp_per_segment, j = 1:4]

    segment_cp_pos = hcat([[1. - i / itp_order, i / itp_order] for i = 1:cp_per_segment]...)
    segment_start_vertex = [1, 2, 4, 1] # Note top/left is reverse

    cp_per_face = 0
    face_cp_ids = zeros(FEM_Int, cp_per_face, 1)
    face_cp_pos = zeros(FEM_Float, 0, cp_per_face)
    face_start_segments = zeros(FEM_Int, 0, 0)

    cp_per_block = 0
    block_cp_ids = zeros(FEM_Int, cp_per_block, 0)
    block_cp_pos = zeros(FEM_Float, 0, cp_per_block)

    return @Construct Classical_Element_Structure
end

function init_Structure_Cube3D_Serendipity(itp_order::Integer)
    cp_per_dim = itp_order + 1

    cp_per_vertex = 1
    vertex_cp_ids = [1 2 4 3 5 6 8 7]

    cp_per_segment = cp_per_dim - 2
    segment_orders = [0, 5, 1, 4, 8, 9, 11, 10, 2, 7, 3, 6]
    segment_cp_ids = [segment_orders[j] * cp_per_segment + 8 + i for i = 1:cp_per_segment, j = 1:12]
    segment_cp_pos = hcat([[1. - i / itp_order, i / itp_order] for i = 1:cp_per_segment]...)
    segment_start_vertex = [1, 2, 4, 1,
    1, 2, 3, 4,
    5, 6, 8, 5]

    cp_per_face = 0
    face_cp_ids = zeros(FEM_Int, cp_per_face, 6)
    face_cp_pos = zeros(FEM_Float, 0, cp_per_face)
    face_start_segments = zeros(FEM_Int, 0, 0)

    cp_per_block = 0
    block_cp_ids = zeros(FEM_Int, cp_per_block, 1)
    block_cp_pos = zeros(FEM_Float, 0, cp_per_block)
    return @Construct Classical_Element_Structure
end
