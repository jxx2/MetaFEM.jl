# Adapted from MacroTool.jl
walk(x, inner, outer) = outer(x)
walk(x::Expr, inner, outer) = outer(Expr(x.head, map(inner, x.args)...))

postwalk(f, x) = walk(x, x -> postwalk(f, x), f)
prewalk(f, x)  = walk(f(x), x -> prewalk(f, x), identity)

isexpr(x::Expr) = true
isexpr(x) = false
isexpr(x::Expr, ts...) = x.head in ts
isexpr(x, ts...) = any(T->isa(T, Type) && isa(x, T), ts)

isline(ex) = isexpr(ex, :line) || isa(ex, LineNumberNode)
iscall(ex, f) = isexpr(ex, :call) && ex.args[1] == f

rmlines(x) = x
function rmlines(x::Expr)
  # Do not strip the first argument to a macrocall, which is required.
  if x.head == :macrocall && length(x.args) >= 2
    Expr(x.head, x.args[1], nothing, filter(x->!isline(x), x.args[3:end])...)
  else
    Expr(x.head, filter(x->!isline(x), x.args)...)
  end
end
striplines(ex) = prewalk(rmlines, ex)

#Personal defination
is_block(ex) = isexpr(ex, :block)
is_collection(ex) = isexpr(ex, :tuple, :vect, :vcat)
rmblock(x) = x
rmblock(x::Expr) = x.head == :block ? Expr(:block, vcat(map(x -> x.head == :block ? x.args : x, x.args)...)...) : x
stripblocks(ex) = postwalk(rmblock, ex)
