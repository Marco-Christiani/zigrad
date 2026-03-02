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
#include "zigrad/ZigradDialect.h"

namespace mlir::zigrad {
namespace {

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

  static PassRegistration<ZigradKernelLegalizePass> pass_registration;

  static PassPipelineRegistration<> pipeline_registration(
      "zg-kernel-legalize-pipeline",
      "Legalize zigrad kernel carrier ops to stablehlo custom_call.",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(std::make_unique<ZigradKernelLegalizePass>());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
      });

  (void)pass_registration;
  (void)pipeline_registration;

  registered = true;
}

} // namespace mlir::zigrad
