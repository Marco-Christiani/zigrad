#include "zigrad/ZigradKernelLegalizePass.h"

#include <memory>
#include <optional>

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

constexpr const char *kDispatchTargetName = "zigrad.kernel.dispatch";
constexpr int kTypedFfiApiVersion = 4;

static bool is_kernelizable_carrier_source(Operation *op) {
    const StringRef op_name = op->getName().getStringRef();
    return op_name == "stablehlo.dot_general" || op_name == "stablehlo.dot";
}

struct KernelMarkers {
    StringAttr provider;
    StringAttr kernel_key;
};

static std::optional<KernelMarkers> resolve_kernel_markers(Operation *op, func::FuncOp func) {
    auto provider = op->getAttrOfType<StringAttr>("zigrad.kernelize.provider");
    if (!provider) {
        provider = func->getAttrOfType<StringAttr>("zigrad.kernelize.provider");
    }
    if (!provider) return std::nullopt;

    auto kernel_key = op->getAttrOfType<StringAttr>("zigrad.kernelize.region");
    if (!kernel_key) {
        kernel_key = func.getSymNameAttr();
    }
    if (!kernel_key) return std::nullopt;

    return KernelMarkers{provider, kernel_key};
}

static bool marker_pair_equal(const KernelMarkers &lhs, const KernelMarkers &rhs) {
    return lhs.provider == rhs.provider && lhs.kernel_key == rhs.kernel_key;
}

static bool is_named_op(Operation *op, StringRef name) {
    return op != nullptr && op->getName().getStringRef() == name;
}

static Operation *create_kernel_call(Operation *anchor,
                                     ValueRange operands,
                                     TypeRange result_types,
                                     const KernelMarkers &markers,
                                     StringRef pattern,
                                     PatternRewriter &rewriter) {
  NamedAttrList backend_fields;
  backend_fields.append("zigrad.kernel_key", markers.kernel_key);
  backend_fields.append("zigrad.provider", markers.provider);
  backend_fields.append("zigrad.pattern", rewriter.getStringAttr(pattern));

  OperationState state(anchor->getLoc(), "zigrad.kernel_call");
  state.addOperands(operands);
  state.addTypes(result_types);
  state.addAttribute("api_version", rewriter.getI32IntegerAttr(kTypedFfiApiVersion));
  state.addAttribute("call_target_name", rewriter.getStringAttr(kDispatchTargetName));
  state.addAttribute("has_side_effect", rewriter.getBoolAttr(false));
  state.addAttribute("backend_config", rewriter.getDictionaryAttr(backend_fields));

  return rewriter.create(state);
}

static LogicalResult rewrite_matmul_add_chain(Operation *add_op,
                                              func::FuncOp func,
                                              PatternRewriter &rewriter) {
  if (add_op->getNumOperands() != 2) return failure();

  const auto add_markers = resolve_kernel_markers(add_op, func);
  if (!add_markers) return failure();

  const Value add_lhs = add_op->getOperand(0);
  const Value add_rhs = add_op->getOperand(1);
  Operation *add_lhs_def = add_lhs.getDefiningOp();
  Operation *add_rhs_def = add_rhs.getDefiningOp();

  Operation *dot_op = nullptr;
  Value passthrough;
  if (is_kernelizable_carrier_source(add_lhs_def)) {
    dot_op = add_lhs_def;
    passthrough = add_rhs;
  } else if (is_kernelizable_carrier_source(add_rhs_def)) {
    dot_op = add_rhs_def;
    passthrough = add_lhs;
  } else {
    return failure();
  }

  if (dot_op->getNumResults() != 1 || !dot_op->getResult(0).hasOneUse()) {
    return failure();
  }

  const auto dot_markers = resolve_kernel_markers(dot_op, func);
  if (dot_markers && !marker_pair_equal(*add_markers, *dot_markers)) {
    return failure();
  }

  if (dot_op->getNumOperands() != 2) return failure();

  SmallVector<Value, 3> call_operands = {
      dot_op->getOperand(0),
      dot_op->getOperand(1),
      passthrough,
  };

  Operation *replacement =
      create_kernel_call(add_op, call_operands, add_op->getResultTypes(),
                         *add_markers, "dot_add", rewriter);
  rewriter.replaceOp(add_op, replacement->getResults());

  if (dot_op->use_empty()) rewriter.eraseOp(dot_op);

  return success();
}

static LogicalResult rewrite_matmul_unary_chain(Operation *unary_op,
                                                func::FuncOp func,
                                                PatternRewriter &rewriter) {
  const StringRef unary_name = unary_op->getName().getStringRef();
  if (unary_name != "stablehlo.log" && unary_name != "stablehlo.exponential") {
    return failure();
  }

  if (unary_op->getNumOperands() != 1) return failure();

  const auto unary_markers = resolve_kernel_markers(unary_op, func);
  if (!unary_markers) return failure();

  Operation *dot_op = unary_op->getOperand(0).getDefiningOp();
  if (!is_kernelizable_carrier_source(dot_op)) return failure();

  if (dot_op->getNumResults() != 1 || !dot_op->getResult(0).hasOneUse()) {
    return failure();
  }

  const auto dot_markers = resolve_kernel_markers(dot_op, func);
  if (dot_markers && !marker_pair_equal(*unary_markers, *dot_markers)) {
    return failure();
  }

  if (dot_op->getNumOperands() != 2) return failure();

  SmallVector<Value, 2> call_operands = {
      dot_op->getOperand(0),
      dot_op->getOperand(1),
  };

  Operation *replacement =
      create_kernel_call(unary_op, call_operands, unary_op->getResultTypes(),
                         *unary_markers,
                         unary_name == "stablehlo.log" ? "dot_log" : "dot_exp",
                         rewriter);
  rewriter.replaceOp(unary_op, replacement->getResults());

  if (dot_op->use_empty()) rewriter.eraseOp(dot_op);

  return success();
}

struct MatmulAddMulToKernelCallPattern final : RewritePattern {
  /// Match stablehlo.multiply(stablehlo.add(stablehlo.dot, x), x) and emit
  /// one zigrad.kernel_call carrier op.
  explicit MatmulAddMulToKernelCallPattern(MLIRContext *ctx)
      : RewritePattern("stablehlo.multiply", 3, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2) return failure();

    auto func = op->getParentOfType<func::FuncOp>();
    if (!func) return failure();

    const auto mul_markers = resolve_kernel_markers(op, func);
    if (!mul_markers) return failure();

    const Value mul_lhs = op->getOperand(0);
    const Value mul_rhs = op->getOperand(1);
    Operation *mul_lhs_def = mul_lhs.getDefiningOp();
    Operation *mul_rhs_def = mul_rhs.getDefiningOp();

    Operation *add_op = nullptr;
    Value passthrough;
    if (is_named_op(mul_lhs_def, "stablehlo.add")) {
      add_op = mul_lhs_def;
      passthrough = mul_rhs;
    } else if (is_named_op(mul_rhs_def, "stablehlo.add")) {
      add_op = mul_rhs_def;
      passthrough = mul_lhs;
    } else {
      return failure();
    }

    const auto add_markers = resolve_kernel_markers(add_op, func);
    if (add_markers && !marker_pair_equal(*mul_markers, *add_markers)) {
      return failure();
    }

    if (add_op->getNumOperands() != 2) return failure();

    const Value add_lhs = add_op->getOperand(0);
    const Value add_rhs = add_op->getOperand(1);
    Operation *add_lhs_def = add_lhs.getDefiningOp();
    Operation *add_rhs_def = add_rhs.getDefiningOp();

    Operation *dot_op = nullptr;
    if (is_kernelizable_carrier_source(add_lhs_def) && add_rhs == passthrough) {
      dot_op = add_lhs_def;
    } else if (is_kernelizable_carrier_source(add_rhs_def) && add_lhs == passthrough) {
      dot_op = add_rhs_def;
    } else {
      return failure();
    }

    const auto dot_markers = resolve_kernel_markers(dot_op, func);
    if (dot_markers && !marker_pair_equal(*mul_markers, *dot_markers)) {
      return failure();
    }

    if (dot_op->getNumOperands() != 2) return failure();

    SmallVector<Value, 3> call_operands = {
        dot_op->getOperand(0),
        dot_op->getOperand(1),
        passthrough,
    };

    Operation *replacement =
        create_kernel_call(op, call_operands, op->getResultTypes(), *mul_markers,
                           "dot_add_mul", rewriter);
    rewriter.replaceOp(op, replacement->getResults());

    if (add_op->use_empty()) rewriter.eraseOp(add_op);
    if (dot_op->use_empty()) rewriter.eraseOp(dot_op);

    return success();
  }
};

static bool has_same_region_consumers(Operation *source_op, func::FuncOp func,
                                      const KernelMarkers &markers) {
    for (Value result : source_op->getResults()) {
        for (OpOperand &use : result.getUses()) {
            auto use_markers = resolve_kernel_markers(use.getOwner(), func);
            if (use_markers && marker_pair_equal(*use_markers, markers)) {
                return true;
            }
        }
    }
    return false;
}

static LogicalResult rewrite_single_source_to_kernel_call(
    Operation *source_op, func::FuncOp func, PatternRewriter &rewriter) {
  const auto markers = resolve_kernel_markers(source_op, func);
  if (!markers) return failure();

  if (has_same_region_consumers(source_op, func, *markers)) {
    return failure();
  }

  Operation *replacement =
      create_kernel_call(source_op, source_op->getOperands(),
                         source_op->getResultTypes(), *markers,
                         source_op->getName().getStringRef() == "stablehlo.dot_general"
                             ? "dot_general"
                             : "dot",
                         rewriter);
  rewriter.replaceOp(source_op, replacement->getResults());
  return success();
}

struct MatmulAddToKernelCallPattern final : RewritePattern {
  /// Match stablehlo.add(stablehlo.dot, x) and emit zigrad.kernel_call.
  explicit MatmulAddToKernelCallPattern(MLIRContext *ctx)
      : RewritePattern("stablehlo.add", 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func) return failure();
    return rewrite_matmul_add_chain(op, func, rewriter);
  }
};

struct MatmulUnaryToKernelCallPattern final : RewritePattern {
  /// Match stablehlo.{log,exponential}(stablehlo.dot) and emit
  /// zigrad.kernel_call.
  explicit MatmulUnaryToKernelCallPattern(MLIRContext *ctx,
                                          StringRef op_name)
      : RewritePattern(op_name, 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func) return failure();
    return rewrite_matmul_unary_chain(op, func, rewriter);
  }
};

struct SourceToKernelCallPattern final : RewritePattern {
  /// Fallback: match a single kernelizable source op with no same-region
  /// downstream consumers and emit zigrad.kernel_call.
  explicit SourceToKernelCallPattern(MLIRContext *ctx, StringRef op_name)
      : RewritePattern(op_name, 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func) return failure();
    return rewrite_single_source_to_kernel_call(op, func, rewriter);
  }
};

struct ZigradKernelSelectPass final
    : PassWrapper<ZigradKernelSelectPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const final { return "zg-kernel-select"; }

  StringRef getDescription() const final {
    return "Select kernelizable StableHLO chains and rewrite to zigrad.kernel_call.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulAddMulToKernelCallPattern>(&getContext());
    patterns.add<MatmulAddToKernelCallPattern>(&getContext());
    patterns.add<MatmulUnaryToKernelCallPattern>(&getContext(),
                                                 "stablehlo.log");
    patterns.add<MatmulUnaryToKernelCallPattern>(&getContext(),
                                                 "stablehlo.exponential");
    patterns.add<SourceToKernelCallPattern>(&getContext(),
                                            "stablehlo.dot_general");
    patterns.add<SourceToKernelCallPattern>(&getContext(), "stablehlo.dot");

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

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

  static PassRegistration<ZigradKernelSelectPass> select_registration;
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
  (void)select_registration;
  (void)pipeline_registration;

  registered = true;
}

} // namespace mlir::zigrad
