MetaFEM.report_memory(ref_mesh)
##
MetaFEM.estimate_memory_GPU(ref_mesh)
##
isstructtype(ref_mesh |> typeof)
##
fieldnames(ref_mesh |> typeof)
##
ref_mesh.vertices

##
isstructtype(ref_mesh.vertices |> typeof)
fieldnames(ref_mesh.vertices |> typeof)
##
data = getfield(ref_mesh.vertices, :data)
##
MetaFEM.estimate_memory_CPU(data)
##
fieldnames(data |> typeof)
##
data.vals .|>e
##
sizeof(data)
##
values(data)
##
sizeof.(values(data))
##
values(data)
##
MetaFEM.estimate_memory_CPU.(values(data))
##
zip([1,2],[2,3])
##

const MEMORY_UNIT = Dict{Symbol, Int}([:B, :KB, :MB, :GB, :TB] .=> [Int(1 << (10 * i)) for i = 0:4])
##
using Plots
##
using CUDA

CUDA.rand(UInt64, 100)

##
using CUDA

IDs = findall(CUDA.ones(Bool, 100))
vals = CUDA.ones(Int64, 100)
B = CUDA.zeros(Int32, 2, 100)
##
B[1, IDs] .= vals
##
MetaFEM.update_K_NonLinear_1