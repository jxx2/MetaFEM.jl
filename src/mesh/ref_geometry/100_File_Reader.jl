"""
    read_Mesh(filename::String)

The function loads an external mesh. Currently supported file types are:
* `.inp` file, Abaqus first order mesh.
* `.mphtxt` file, COMSOL mesh.
"""
function read_Mesh(filename::String) #read coor and connection to form ref geometry
    tail_name = match(r"\.(?<extension>[a-z|A-Z|0-9]*$)", filename)[:extension]

    if lowercase(tail_name) == "inp"
        coors, connections = open(read_INP, filename, "r")
    elseif lowercase(tail_name) == "mphtxt"
        coors, connections = open(read_MPHTXT, filename, "r")
    else
        error("Undefined file type")
    end
    return (coors, connections) .|> cu
end

function read_NextLine(is_Comment, io) 
    eof(io) && error("Not enough data")
    next_line = readline(io)
    is_Comment(next_line) ? read_NextLine(is_Comment, io) : return next_line
end

function read_Lines_Until(is_Comment, is_Final, io)
    buffer = String[]
    while true
        next_line = read_NextLine(is_Comment, io) 
        is_Final(next_line) && return (buffer, next_line)
        push!(buffer, next_line)
    end
end
