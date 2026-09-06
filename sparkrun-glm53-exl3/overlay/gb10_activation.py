"""SM121 sigmoid reciprocal with one FP32 Newton correction."""
import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def reciprocal(value: cutlass.Float32, *, loc=None, ip=None):
    # The caller restricts this to normal positive denominators. Keep the
    # correction fused, including its cancellation, instead of using mul/sub.
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .f32 r, err, neg_value; rcp.approx.ftz.f32 r, $1; "
        "neg.f32 neg_value, $1; "
        "fma.rn.f32 err, neg_value, r, 0f3f800000; fma.rn.f32 $0, err, r, r; }",
        "=f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))
