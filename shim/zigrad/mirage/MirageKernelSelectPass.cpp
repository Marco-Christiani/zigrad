#include "zigrad/mirage/MirageKernelSelectPass.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "zigrad/ZigradKernelUtils.h"

namespace mlir::zigrad::mirage {
namespace {

constexpr StringLiteral kProvider("mirage");

/// Per-pass kernel key counter, reset at each pass invocation. Patterns hold a
/// pointer to the counter owned by MirageKernelSelectPass::runOnOperation so
/// keys start from mk_0 for every compilation unit.
struct KeyCounter {
  unsigned value = 0;
  std::string next() { return "mk_" + std::to_string(value++); }
};

// ============================================================================
// Pattern: mul(add(dot, x), x) -> "dot_add_mul"  (priority 3)
// ============================================================================

struct DotAddMulPattern final : RewritePattern {
  KeyCounter *counter;

  explicit DotAddMulPattern(MLIRContext *ctx, KeyCounter *counter)
      : RewritePattern("stablehlo.multiply", 3, ctx), counter(counter) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2) return failure();

    const Value mul_lhs = op->getOperand(0);
    const Value mul_rhs = op->getOperand(1);
    Operation *mul_lhs_def = mul_lhs.getDefiningOp();
    Operation *mul_rhs_def = mul_rhs.getDefiningOp();

    Operation *add_op = nullptr;
    Value mul_passthrough;
    if (kernel_utils::is_named_op(mul_lhs_def, "stablehlo.add")) {
      add_op = mul_lhs_def;
      mul_passthrough = mul_rhs;
    } else if (kernel_utils::is_named_op(mul_rhs_def, "stablehlo.add")) {
      add_op = mul_rhs_def;
      mul_passthrough = mul_lhs;
    } else {
      return failure();
    }

    if (add_op->getNumOperands() != 2) return failure();

    const Value add_lhs = add_op->getOperand(0);
    const Value add_rhs = add_op->getOperand(1);
    Operation *add_lhs_def = add_lhs.getDefiningOp();
    Operation *add_rhs_def = add_rhs.getDefiningOp();

    Operation *dot_op = nullptr;
    if (kernel_utils::is_dot_op(add_lhs_def) && add_rhs == mul_passthrough) {
      dot_op = add_lhs_def;
    } else if (kernel_utils::is_dot_op(add_rhs_def) && add_lhs == mul_passthrough) {
      dot_op = add_rhs_def;
    } else {
      return failure();
    }

    if (dot_op->getNumResults() != 1 || !dot_op->getResult(0).hasOneUse())
      return failure();
    if (dot_op->getNumOperands() != 2) return failure();

    SmallVector<Value, 3> call_operands = {
        dot_op->getOperand(0),
        dot_op->getOperand(1),
        mul_passthrough,
    };

    std::string key = counter->next();
    Operation *replacement = kernel_utils::create_kernel_call(
        op, call_operands, op->getResultTypes(),
        kProvider, key, "dot_add_mul", rewriter);
    rewriter.replaceOp(op, replacement->getResults());

    if (add_op->use_empty()) rewriter.eraseOp(add_op);
    if (dot_op->use_empty()) rewriter.eraseOp(dot_op);

    return success();
  }
};

// ============================================================================
// Pattern: add(dot, x) -> "dot_add"  (priority 2)
// ============================================================================

struct DotAddPattern final : RewritePattern {
  KeyCounter *counter;

  explicit DotAddPattern(MLIRContext *ctx, KeyCounter *counter)
      : RewritePattern("stablehlo.add", 2, ctx), counter(counter) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2) return failure();

    const Value lhs = op->getOperand(0);
    const Value rhs = op->getOperand(1);
    Operation *lhs_def = lhs.getDefiningOp();
    Operation *rhs_def = rhs.getDefiningOp();

    Operation *dot_op = nullptr;
    Value passthrough;
    if (kernel_utils::is_dot_op(lhs_def)) {
      dot_op = lhs_def;
      passthrough = rhs;
    } else if (kernel_utils::is_dot_op(rhs_def)) {
      dot_op = rhs_def;
      passthrough = lhs;
    } else {
      return failure();
    }

    if (dot_op->getNumResults() != 1 || !dot_op->getResult(0).hasOneUse())
      return failure();
    if (dot_op->getNumOperands() != 2) return failure();

    SmallVector<Value, 3> call_operands = {
        dot_op->getOperand(0),
        dot_op->getOperand(1),
        passthrough,
    };

    std::string key = counter->next();
    Operation *replacement = kernel_utils::create_kernel_call(
        op, call_operands, op->getResultTypes(),
        kProvider, key, "dot_add", rewriter);
    rewriter.replaceOp(op, replacement->getResults());

    if (dot_op->use_empty()) rewriter.eraseOp(dot_op);

    return success();
  }
};

// ============================================================================
// Pattern: exp(dot) -> "dot_exp",  log(dot) -> "dot_log"  (priority 2)
// ============================================================================

struct DotUnaryPattern final : RewritePattern {
  KeyCounter *counter;

  explicit DotUnaryPattern(MLIRContext *ctx, StringRef op_name,
                           KeyCounter *counter)
      : RewritePattern(op_name, 2, ctx), counter(counter) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1) return failure();

    Operation *dot_op = op->getOperand(0).getDefiningOp();
    if (!kernel_utils::is_dot_op(dot_op)) return failure();

    if (dot_op->getNumResults() != 1 || !dot_op->getResult(0).hasOneUse())
      return failure();
    if (dot_op->getNumOperands() != 2) return failure();

    SmallVector<Value, 2> call_operands = {
        dot_op->getOperand(0),
        dot_op->getOperand(1),
    };

    const StringRef unary_name = op->getName().getStringRef();
    const StringRef pattern =
        unary_name == "stablehlo.log" ? "dot_log" : "dot_exp";

    std::string key = counter->next();
    Operation *replacement = kernel_utils::create_kernel_call(
        op, call_operands, op->getResultTypes(),
        kProvider, key, pattern, rewriter);
    rewriter.replaceOp(op, replacement->getResults());

    if (dot_op->use_empty()) rewriter.eraseOp(dot_op);

    return success();
  }
};

// ============================================================================
// Pass definition
// ============================================================================

struct MirageKernelSelectPass final
    : PassWrapper<MirageKernelSelectPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const final { return "zg-mirage-kernel-select"; }

  StringRef getDescription() const final {
    return "Select fuseable StableHLO patterns for the Mirage kernel provider.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    KeyCounter counter;

    RewritePatternSet patterns(&getContext());
    patterns.add<DotAddMulPattern>(&getContext(), &counter);
    patterns.add<DotAddPattern>(&getContext(), &counter);
    patterns.add<DotUnaryPattern>(&getContext(), "stablehlo.exponential",
                                  &counter);
    patterns.add<DotUnaryPattern>(&getContext(), "stablehlo.log", &counter);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void registerMirageKernelSelectPass() {
  static bool registered = false;
  if (registered) return;

  static PassRegistration<MirageKernelSelectPass> registration;
  (void)registration;

  registered = true;
}

} // namespace mlir::zigrad::mirage
