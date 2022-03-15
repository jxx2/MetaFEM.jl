is_Comment_INP(x::String) = ~isnothing(match(r"^\*\*", x))
is_Keyword_INP(x::String) = ~isnothing(match(r"^\*", x))
has_Nextline_INP(x::String) = ~isnothing(match(r", *$", x)) # One logical sentence in multiple lines separated by ","

collect_Vec_INP(x::String) = split(strip(isspace, x), ", ")
block_Finished(x) = isempty(x) || is_Keyword_INP(x)
get_Block_Content_INP(io) = read_Lines_Until(is_Comment_INP, block_Finished, io)

function read_INP(io)
    flags = zeros(Bool, 2) #Node and element
    mesh_data = Dict()
    this_line = read_NextLine(is_Comment_INP, io)

    while true
        is_Keyword_INP(this_line) || error(".inp file parsing error, line {$header} should be a keyword")
        while has_Nextline_INP(this_line)
            this_line = string(this_line, readline(io))
        end
        header_contents = collect_Vec_INP(this_line)

        if uppercase(header_contents[1]) == "*NODE"
            buffer, this_line = get_Block_Content_INP(io)
            buffer = hcat(collect_Vec_INP.(buffer)...)

            mesh_data[:vids] = parse.(FEM_Int, vec(buffer[1, :]))
            mesh_data[:coors] = parse.(FEM_Float, buffer[2:end, :])

            println("This block is a ", "NODE")
            flags[1] = true
        elseif uppercase(header_contents[1]) == "*ELEMENT"
            buffer, this_line = get_Block_Content_INP(io)
            buffer = hcat(collect_Vec_INP.(buffer)...)

            # mesh_data[:elids] = parse.(FEM_Int, vec(buffer[1, :]))
            mesh_data[:el_vids] = parse.(FEM_Int, buffer[2:end, :])

            println("This block is an ", "ELEMENT")
            flags[2] = true
        else
            buffer, this_line = get_Block_Content_INP(io)
            println("This block is something else.")
        end
        
        prod(flags) || continue
        
        vids = mesh_data[:vids]
        coors = mesh_data[:coors]
        el_vids = mesh_data[:el_vids]

        local_connections = zeros(FEM_Int, maximum(vids))
        local_vids = findall(ones(Bool, length(vids)))
        local_connections[vids] .= local_vids #replace the IDs in connections to local 
        return coors, local_connections[el_vids]
    end
end