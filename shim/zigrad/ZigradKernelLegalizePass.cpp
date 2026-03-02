#include "zigrad/ZigradKernelLegalizePass.h"

#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/CAPI/IR.h"
#include "stablehlo/integrations/c/StablehloAttributes.h"
#include "zigrad/ZigradDialect.h"

namespace mlir::zigrad {
namespace {

// ============================================================================
// Expand pass: revert kernel_call ops back to original StableHLO ops.
//
// Used when a provider returns Unsupported during materialization: the
// kernel_call is expanded back to the StableHLO pattern it was created from
// (dot_add → add(dot(a,b), c), etc.) so the backend can handle it natively.
// ============================================================================

static StringRef get_backend_config_string(Operation *op, StringRef key) {
  auto bc = op->getAttrOfType<DictionaryAttr>("backend_config");
  if (!bc) return {};
  auto attr = bc.getAs<StringAttr>(key);
  if (!attr) return {};
  return attr.getValue();
}

/// Build a stablehlo.dot_general op with standard matmul dimension numbers
/// derived from the LHS/RHS tensor ranks. Contracting dimension is the last
/// axis of LHS against the first axis (dim 0) of RHS; no batching dimensions.
static Operation *create_dot_general(Location loc, Value lhs, Value rhs,
                                      TypeRange result_types,
                                      PatternRewriter &rewriter) {
  auto lhs_ranked = cast<RankedTensorType>(lhs.getType());
  int64_t lhs_contract = lhs_ranked.getRank() - 1;
  int64_t rhs_contract = 0;

  // Build the #stablehlo.dot attribute via C API.
  MlirContext capi_ctx = wrap(rewriter.getContext());
  MlirAttribute capi_attr = stablehloDotDimensionNumbersGet(
      capi_ctx,
      /*nLhsBatchingDimensions=*/0, /*lhsBatchingDimensions=*/nullptr,
      /*nRhsBatchingDimensions=*/0, /*rhsBatchingDimensions=*/nullptr,
      /*nLhsContractingDimensions=*/1, /*lhsContractingDimensions=*/&lhs_contract,
      /*nRhsContractingDimensions=*/1, /*rhsContractingDimensions=*/&rhs_contract);
  Attribute dot_dims_attr = unwrap(capi_attr);

  OperationState state(loc, "stablehlo.dot_general");
  state.addOperands({lhs, rhs});
  state.addTypes(result_types);
  state.addAttribute("dot_dimension_numbers", dot_dims_attr);

  return rewriter.create(state);
}

struct KernelCallExpandPattern final : OpRewritePattern<KernelCallOp> {
  using OpRewritePattern<KernelCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(KernelCallOp op,
                                PatternRewriter &rewriter) const override {
    StringRef pattern = get_backend_config_string(op, "zigrad.pattern");
    if (pattern.empty()) return failure();

    auto inputs = op.getInputs();

    if (pattern == "dot_add" && inputs.size() == 3) {
      Operation *dot = create_dot_general(op.getLoc(), inputs[0], inputs[1],
                                          op->getResultTypes(), rewriter);
      if (!dot) return failure();

      OperationState add_state(op.getLoc(), "stablehlo.add");
      add_state.addOperands({dot->getResult(0), inputs[2]});
      add_state.addTypes(op->getResultTypes());
      Operation *add = rewriter.create(add_state);
      rewriter.replaceOp(op, add->getResults());
      return success();
    }

    if ((pattern == "dot_exp" || pattern == "dot_log") && inputs.size() == 2) {
      Operation *dot = create_dot_general(op.getLoc(), inputs[0], inputs[1],
                                          op->getResultTypes(), rewriter);
      if (!dot) return failure();

      StringRef unary_name = pattern == "dot_exp" ? "stablehlo.exponential"
                                                  : "stablehlo.log";
      OperationState unary_state(op.getLoc(), unary_name);
      unary_state.addOperands(dot->getResult(0));
      unary_state.addTypes(op->getResultTypes());
      Operation *unary = rewriter.create(unary_state);
      rewriter.replaceOp(op, unary->getResults());
      return success();
    }

    if (pattern == "dot_add_mul" && inputs.size() == 3) {
      Operation *dot = create_dot_general(op.getLoc(), inputs[0], inputs[1],
                                          op->getResultTypes(), rewriter);
      if (!dot) return failure();

      OperationState add_state(op.getLoc(), "stablehlo.add");
      add_state.addOperands({dot->getResult(0), inputs[2]});
      add_state.addTypes(op->getResultTypes());
      Operation *add = rewriter.create(add_state);

      OperationState mul_state(op.getLoc(), "stablehlo.multiply");
      mul_state.addOperands({add->getResult(0), inputs[2]});
      mul_state.addTypes(op->getResultTypes());
      Operation *mul = rewriter.create(mul_state);
      rewriter.replaceOp(op, mul->getResults());
      return success();
    }

    if ((pattern == "dot" || pattern == "dot_general") && inputs.size() == 2) {
      Operation *dot = create_dot_general(op.getLoc(), inputs[0], inputs[1],
                                          op->getResultTypes(), rewriter);
      if (!dot) return failure();
      rewriter.replaceOp(op, dot->getResults());
      return success();
    }

    return failure();
  }
};

struct ZigradKernelCallExpandPass final
    : PassWrapper<ZigradKernelCallExpandPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const final { return "zg-kernel-call-expand"; }

  StringRef getDescription() const final {
    return "Expand un-materialized zigrad.kernel_call ops back to StableHLO.";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<KernelCallExpandPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// ============================================================================
// Legalize pass: kernel_call → stablehlo.custom_call
// ============================================================================

static FailureOr<ArrayAttr> build_default_layouts_for_values(PatternRewriter &rewriter,
                                                              ValueRange values) {
  SmallVector<Attribute> layouts;
  layouts.reserve(values.size());

  for (Value value : values) {
    auto ranked = dyn_cast<RankedTensorType>(value.getType());
    if (!ranked) return failure();

    const int64_t rank = ranked.getRank();
    llvm::SmallVector<int64_t> order(rank);
    for (int64_t i = 0; i < rank; ++i) {
      order[i] = rank - i - 1;
    }

    auto layout_ty = RankedTensorType::get({rank}, rewriter.getIndexType());
    layouts.push_back(DenseIntElementsAttr::get(layout_ty, order));
  }

  return rewriter.getArrayAttr(layouts);
}

static FailureOr<ArrayAttr> build_default_layouts_for_types(PatternRewriter &rewriter,
                                                             TypeRange types) {
  SmallVector<Attribute> layouts;
  layouts.reserve(types.size());

  for (Type type : types) {
    auto ranked = dyn_cast<RankedTensorType>(type);
    if (!ranked) return failure();

    const int64_t rank = ranked.getRank();
    llvm::SmallVector<int64_t> order(rank);
    for (int64_t i = 0; i < rank; ++i) {
      order[i] = rank - i - 1;
    }

    auto layout_ty = RankedTensorType::get({rank}, rewriter.getIndexType());
    layouts.push_back(DenseIntElementsAttr::get(layout_ty, order));
  }

  return rewriter.getArrayAttr(layouts);
}

struct KernelCallToStablehloCustomCallPattern final : OpRewritePattern<KernelCallOp> {
  using OpRewritePattern<KernelCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(KernelCallOp op, PatternRewriter &rewriter) const override {
    auto operand_layouts = build_default_layouts_for_values(rewriter, op.getInputs());
    if (failed(operand_layouts)) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor operands");
    }

    auto result_layouts = build_default_layouts_for_types(rewriter, op->getResultTypes());
    if (failed(result_layouts)) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor results");
    }

    OperationState state(op.getLoc(), "stablehlo.custom_call");
    state.addOperands(op.getInputs());
    state.addTypes(op->getResultTypes());
    state.addAttribute("api_version", op.getApiVersionAttr());
    state.addAttribute("call_target_name", op.getCallTargetNameAttr());
    state.addAttribute("has_side_effect", op.getHasSideEffectAttr());
    state.addAttribute("backend_config", op.getBackendConfigAttr());
    state.addAttribute("operand_layouts", *operand_layouts);
    state.addAttribute("result_layouts", *result_layouts);
    state.addAttribute("output_operand_aliases", rewriter.getArrayAttr({}));

    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(op, replacement->getResults());
    return success();
  }
};

struct ZigradKernelLegalizePass final
    : PassWrapper<ZigradKernelLegalizePass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const final { return "zg-kernel-legalize"; }

  StringRef getDescription() const final {
    return "Legalize zigrad.kernel_call operations to stablehlo.custom_call.";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<KernelCallToStablehloCustomCallPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void registerZigradKernelLegalizePasses() {
  static bool registered = false;
  if (registered) return;

  static PassRegistration<ZigradKernelCallExpandPass> expand_registration;
  static PassRegistration<ZigradKernelLegalizePass> pass_registration;

  static PassPipelineRegistration<> pipeline_registration(
      "zg-kernel-legalize-pipeline",
      "Legalize zigrad kernel carrier ops to stablehlo custom_call.",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(std::make_unique<ZigradKernelLegalizePass>());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
      });

  (void)expand_registration;
  (void)pass_registration;
  (void)pipeline_registration;

  registered = true;
}

} // namespace mlir::zigrad
