is_Comment_MPHTXT(x::String) = ~isnothing(match(r"^#", x)) || isempty(x)
collect_Vec_MPHTXT(x::String) = split(strip(isspace, x), " ")

function read_MPHTXT(io)
    flags = zeros(Bool, 2) #Node and element
    mesh_data = Dict()

    while true
        this_line = read_NextLine(is_Comment_MPHTXT, io)

        line_data = collect_Vec_MPHTXT(this_line)
        if length(line_data) >= 6 && prod(line_data[3:6] .== collect_Vec_MPHTXT("number of mesh points"))
            vertex_num = parse(Int, line_data[1])
            this_line = read_NextLine(is_Comment_MPHTXT, io)
            mesh_data[:start_vid] = parse(FEM_Int, collect_Vec_MPHTXT(this_line)[1])

            first_coor = parse.(FEM_Float, collect_Vec_MPHTXT(read_NextLine(is_Comment_MPHTXT, io)))
            coors = mesh_data[:coors] = zeros(FEM_Float, length(first_coor), vertex_num)
            coors[:, 1] .= first_coor
            for i = 2:vertex_num
                coors[:, i] .= parse.(FEM_Float, collect_Vec_MPHTXT(read_NextLine(is_Comment_MPHTXT, io)))
            end

            println("Read ", "NODE")
            flags[1] = true
        elseif length(line_data) >= 5 && prod(line_data[3:5] .== collect_Vec_MPHTXT("number of elements"))
            element_num = parse(Int, line_data[1])

            first_connection = parse.(FEM_Int, collect_Vec_MPHTXT(read_NextLine(is_Comment_MPHTXT, io)))
            connection = mesh_data[:connection] = zeros(FEM_Float, length(first_connection), element_num)
            connection[:, 1] .= first_connection 
            for i = 2:element_num
                connection[:, i] .= parse.(FEM_Int, collect_Vec_MPHTXT(read_NextLine(is_Comment_MPHTXT, io)))
            end

            println("Read ", "Element")
            flags[2] = true
        end

        prod(flags) || continue
        return mesh_data[:coors], mesh_data[:connection] .- (mesh_data[:start_vid] - 1)
    end
end