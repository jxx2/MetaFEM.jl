# for derivative
@Define_Semantic_Constraint S₁ ∊ (S₁ isa SymbolicWord)
@Define_Semantic_Constraint S₂ ∊ (S₂ isa SymbolicWord)

@Define_Semantic_Constraint f_undef ∊ (~(f_undef in [:+; :-; :*; :/; :^; :log; :inv]))

@Define_Rewrite_Rule Basic_Diff1 ≔ ∂(S₁, S₁) => 1
@Define_Rewrite_Rule Basic_Diff2 ≔ ∂(S₂, S₁) => 0
@Define_Rewrite_Rule Basic_Diff3 ≔ ∂(N₁, S₁) => 0

@Define_Rewrite_Rule Basic_Diff4 ≔ ∂({f_undef}(a...), S₁) => 0

@Define_Rewrite_Rule Spatial_Diff1 ≔  ∂(N₁, a) => 0
@Define_Rewrite_Rule Spatial_Diff2 ≔  ∂({f_undef}(a...), a) => 0

@Define_Rewrite_Rule Add_Diff ≔ ∂(a + (b...), S₁) => ∂(a, S₁) + ∂(+(b...), S₁)
@Define_Rewrite_Rule Mul_Diff ≔ ∂(a * (b...), S₁) => ∂(a, S₁) * (b...) + a * ∂(*(b...), S₁)
@Define_Rewrite_Rule Pow_Diff ≔ ∂(a ^ b, S₁) => ∂(a, S₁) * a ^ (b - 1) * b  + ∂(b, S₁) * log(a) * a ^ b

Basic_Diff_Rules = [Basic_Diff1, Basic_Diff2, Basic_Diff3, Basic_Diff4]
Spatial_Diff_Rules = [Spatial_Diff1, Spatial_Diff2]

Diff_Add_Rules = [Add_Diff]
Diff_Mul_Rules = [Mul_Diff]
Diff_Pow_Rules = [Pow_Diff]
Diff_Operator_Rules = [Add_Diff, Mul_Diff, Pow_Diff]

@Compile_Rewrite_Rules diff_Operator Diff_Operator_Rules
@Compile_Rewrite_Rules diff_Base Basic_Diff_Rules
@Compile_Rewrite_Rules diff_Spatial Spatial_Diff_Rules

function do_SymbolicDiff(source_term::GroundTerm, variable::SymbolicWord)
    diff_term = construct_Term(:∂, [source_term; variable])
    while true
        changed_op, diff_term = diff_Operator(diff_term)
        _, diff_term = diff_Base(diff_term)
        diff_term = simplify_Merge(diff_term)
        ~(changed_op) && return diff_term
    end
end
