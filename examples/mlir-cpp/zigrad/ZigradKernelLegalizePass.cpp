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
#include "stablehlo/integrations/c/StablehloDialect.h"
#include "zigrad/ZigradDialect.h"

namespace mlir::zigrad {
namespace {

// Expands a rejected kernel carrier into the StableHLO pattern it replaced.

static StringRef get_backend_config_string(Operation *op, StringRef key) {
  auto bc = op->getAttrOfType<DictionaryAttr>("backend_config");
  if (!bc) return {};
  auto attr = bc.getAs<StringAttr>(key);
  if (!attr) return {};
  return attr.getValue();
}

static FailureOr<int64_t> get_backend_config_int(Operation *op, StringRef key) {
  auto bc = op->getAttrOfType<DictionaryAttr>("backend_config");
  if (!bc) return failure();
  auto attr = bc.getAs<IntegerAttr>(key);
  if (!attr) return failure();
  return attr.getInt();
}

static FailureOr<double> get_backend_config_float(Operation *op,
                                                  StringRef key) {
  auto bc = op->getAttrOfType<DictionaryAttr>("backend_config");
  if (!bc) return failure();
  auto attr = bc.getAs<FloatAttr>(key);
  if (!attr) return failure();
  return attr.getValueAsDouble();
}

static Attribute get_backend_config_attr(Operation *op, StringRef key) {
  auto bc = op->getAttrOfType<DictionaryAttr>("backend_config");
  if (!bc) return {};
  return bc.get(key);
}

/// Creates a single-dimension StableHLO sum reduction.
///
///  The initial value is a scalar zero of the element type.
static Operation *create_reduce_sum(Location loc, Value input, int64_t dim,
                                     PatternRewriter &rewriter) {
  auto input_type = cast<RankedTensorType>(input.getType());
  Type element_type = input_type.getElementType();

  auto zero_type = RankedTensorType::get({}, element_type);
  auto zero_attr = DenseElementsAttr::get(zero_type, rewriter.getZeroAttr(element_type));
  OperationState zero_state(loc, "stablehlo.constant");
  zero_state.addTypes(zero_type);
  zero_state.addAttribute("value", zero_attr);
  Operation *zero_op = rewriter.create(zero_state);

  SmallVector<int64_t> result_shape;
  for (int64_t i = 0; i < input_type.getRank(); ++i) {
    if (i != dim) result_shape.push_back(input_type.getDimSize(i));
  }
  auto result_type = RankedTensorType::get(result_shape, element_type);

  OperationState state(loc, "stablehlo.reduce");
  state.addOperands({input, zero_op->getResult(0)});
  state.addTypes(result_type);
  state.addAttribute("dimensions", rewriter.getDenseI64ArrayAttr({dim}));

  Region *body = state.addRegion();
  Block *block = new Block();
  body->push_back(block);
  auto scalar_type = RankedTensorType::get({}, element_type);
  block->addArgument(scalar_type, loc);
  block->addArgument(scalar_type, loc);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    OperationState add_state(loc, "stablehlo.add");
    add_state.addOperands({block->getArgument(0), block->getArgument(1)});
    add_state.addTypes(scalar_type);
    Operation *add_op = rewriter.create(add_state);
    OperationState ret_state(loc, "stablehlo.return");
    ret_state.addOperands(add_op->getResult(0));
    rewriter.create(ret_state);
  }

  return rewriter.create(state);
}

/// Broadcasts a value into the requested ranked tensor type.
static Operation *create_broadcast_in_dim(Location loc, Value input,
                                           ArrayRef<int64_t> broadcast_dims,
                                           RankedTensorType result_type,
                                           PatternRewriter &rewriter) {
  OperationState state(loc, "stablehlo.broadcast_in_dim");
  state.addOperands(input);
  state.addTypes(result_type);
  state.addAttribute("broadcast_dimensions",
                     rewriter.getDenseI64ArrayAttr(broadcast_dims));
  return rewriter.create(state);
}

static FailureOr<Value> create_rms_norm(Location loc, Value input,
                                        int64_t normalized_size,
                                        PatternRewriter &rewriter) {
  auto input_type = dyn_cast<RankedTensorType>(input.getType());
  if (!input_type || input_type.getRank() < 1 || normalized_size <= 0)
    return failure();

  Type element_type = input_type.getElementType();
  if (!isa<FloatType>(element_type)) return failure();

  const int64_t reduction_dim = input_type.getRank() - 1;
  if (!input_type.isDynamicDim(reduction_dim) &&
      input_type.getDimSize(reduction_dim) != normalized_size)
    return failure();

  OperationState square_state(loc, "stablehlo.multiply");
  square_state.addOperands({input, input});
  square_state.addTypes(input_type);
  Operation *square = rewriter.create(square_state);

  Operation *sum =
      create_reduce_sum(loc, square->getResult(0), reduction_dim, rewriter);
  auto sum_type = cast<RankedTensorType>(sum->getResult(0).getType());
  auto scalar_type = RankedTensorType::get({}, element_type);

  const double scale = 1.0 / static_cast<double>(normalized_size);
  auto scale_attr = DenseElementsAttr::get(
      scalar_type, rewriter.getFloatAttr(element_type, scale));
  OperationState scale_state(loc, "stablehlo.constant");
  scale_state.addTypes(scalar_type);
  scale_state.addAttribute("value", scale_attr);
  Operation *scale_constant = rewriter.create(scale_state);

  SmallVector<int64_t> scalar_broadcast_dims;
  Operation *scale_broadcast = create_broadcast_in_dim(
      loc, scale_constant->getResult(0), scalar_broadcast_dims, sum_type,
      rewriter);

  OperationState mean_state(loc, "stablehlo.multiply");
  mean_state.addOperands(
      {sum->getResult(0), scale_broadcast->getResult(0)});
  mean_state.addTypes(sum_type);
  Operation *mean = rewriter.create(mean_state);

  auto epsilon_attr = DenseElementsAttr::get(
      scalar_type, rewriter.getFloatAttr(element_type, 1.0e-5));
  OperationState epsilon_state(loc, "stablehlo.constant");
  epsilon_state.addTypes(scalar_type);
  epsilon_state.addAttribute("value", epsilon_attr);
  Operation *epsilon_constant = rewriter.create(epsilon_state);
  Operation *epsilon_broadcast = create_broadcast_in_dim(
      loc, epsilon_constant->getResult(0), scalar_broadcast_dims, sum_type,
      rewriter);

  OperationState denominator_state(loc, "stablehlo.add");
  denominator_state.addOperands(
      {mean->getResult(0), epsilon_broadcast->getResult(0)});
  denominator_state.addTypes(sum_type);
  Operation *denominator = rewriter.create(denominator_state);

  OperationState inverse_state(loc, "stablehlo.rsqrt");
  inverse_state.addOperands(denominator->getResult(0));
  inverse_state.addTypes(sum_type);
  Operation *inverse = rewriter.create(inverse_state);

  SmallVector<int64_t> inverse_broadcast_dims;
  for (int64_t dim = 0; dim < input_type.getRank(); ++dim) {
    if (dim != reduction_dim) inverse_broadcast_dims.push_back(dim);
  }
  Operation *inverse_broadcast = create_broadcast_in_dim(
      loc, inverse->getResult(0), inverse_broadcast_dims, input_type, rewriter);

  OperationState result_state(loc, "stablehlo.multiply");
  result_state.addOperands({input, inverse_broadcast->getResult(0)});
  result_state.addTypes(input_type);
  return rewriter.create(result_state)->getResult(0);
}

/// Creates a single-dimension StableHLO maximum reduction.
///
///  The initial value is negative infinity in the element type.
static Operation *create_reduce_max(Location loc, Value input, int64_t dim,
                                     PatternRewriter &rewriter) {
  auto input_type = cast<RankedTensorType>(input.getType());
  Type element_type = input_type.getElementType();

  auto scalar_type = RankedTensorType::get({}, element_type);
  auto neg_inf = APFloat::getInf(
      cast<FloatType>(element_type).getFloatSemantics(), /*Negative=*/true);
  auto init_attr = DenseElementsAttr::get(scalar_type,
                                            rewriter.getFloatAttr(element_type, neg_inf));
  OperationState init_state(loc, "stablehlo.constant");
  init_state.addTypes(scalar_type);
  init_state.addAttribute("value", init_attr);
  Operation *init_op = rewriter.create(init_state);

  SmallVector<int64_t> result_shape;
  for (int64_t i = 0; i < input_type.getRank(); ++i) {
    if (i != dim) result_shape.push_back(input_type.getDimSize(i));
  }
  auto result_type = RankedTensorType::get(result_shape, element_type);

  OperationState state(loc, "stablehlo.reduce");
  state.addOperands({input, init_op->getResult(0)});
  state.addTypes(result_type);
  state.addAttribute("dimensions", rewriter.getDenseI64ArrayAttr({dim}));

  Region *body = state.addRegion();
  Block *block = new Block();
  body->push_back(block);
  block->addArgument(scalar_type, loc);
  block->addArgument(scalar_type, loc);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    OperationState max_state(loc, "stablehlo.maximum");
    max_state.addOperands({block->getArgument(0), block->getArgument(1)});
    max_state.addTypes(scalar_type);
    Operation *max_op = rewriter.create(max_state);
    OperationState ret_state(loc, "stablehlo.return");
    ret_state.addOperands(max_op->getResult(0));
    rewriter.create(ret_state);
  }

  return rewriter.create(state);
}

static Operation *create_dot_general_with_dims(Location loc, Value lhs, Value rhs,
                                                Attribute dot_dims_attr,
                                                TypeRange result_types,
                                                PatternRewriter &rewriter) {
  OperationState state(loc, "stablehlo.dot_general");
  state.addOperands({lhs, rhs});
  state.addTypes(result_types);
  state.addAttribute("dot_dimension_numbers", dot_dims_attr);
  return rewriter.create(state);
}

/// Creates an unbatched matmul using the last LHS and first RHS dimensions.
static Operation *create_dot_general(Location loc, Value lhs, Value rhs,
                                      TypeRange result_types,
                                      PatternRewriter &rewriter) {
  auto lhs_ranked = cast<RankedTensorType>(lhs.getType());
  int64_t lhs_contract = lhs_ranked.getRank() - 1;
  int64_t rhs_contract = 0;

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

    if (pattern == "rms_norm" && inputs.size() == 1) {
      auto normalized_size =
          get_backend_config_int(op, "zigrad.normalized_size");
      if (failed(normalized_size)) return failure();

      auto normalized = create_rms_norm(op.getLoc(), inputs[0],
                                        *normalized_size, rewriter);
      if (failed(normalized)) return failure();
      rewriter.replaceOp(op, *normalized);
      return success();
    }

    if (pattern == "rms_norm_matmul" && inputs.size() == 2) {
      auto normalized_size =
          get_backend_config_int(op, "zigrad.normalized_size");
      if (failed(normalized_size)) return failure();

      auto normalized = create_rms_norm(op.getLoc(), inputs[0],
                                        *normalized_size, rewriter);
      if (failed(normalized)) return failure();

      Operation *result = create_dot_general(
          op.getLoc(), *normalized, inputs[1], op->getResultTypes(), rewriter);
      if (!result) return failure();
      rewriter.replaceOp(op, result->getResults());
      return success();
    }

    if (pattern == "softmax_matmul" && inputs.size() == 2) {
      Location loc = op.getLoc();
      Value scores = inputs[0];
      Value v = inputs[1];
      auto reduction_dim =
          get_backend_config_int(op, "zigrad.reduction_dim");
      Attribute value_dot_dims = get_backend_config_attr(op, "zigrad.value_dot_dims");
      auto scores_type = dyn_cast<RankedTensorType>(scores.getType());
      auto value_type = dyn_cast<RankedTensorType>(v.getType());
      if (failed(reduction_dim) || !value_dot_dims || !scores_type ||
          !value_type || *reduction_dim < 0 ||
          *reduction_dim >= scores_type.getRank())
        return failure();

      OperationState exp_state(loc, "stablehlo.exponential");
      exp_state.addOperands(scores);
      exp_state.addTypes(scores_type);
      Operation *exp_result = rewriter.create(exp_state);

      Operation *sum = create_reduce_sum(loc, exp_result->getResult(0),
                                         *reduction_dim, rewriter);

      SmallVector<int64_t> broadcast_dims;
      for (int64_t i = 0; i < scores_type.getRank(); ++i) {
        if (i != *reduction_dim) broadcast_dims.push_back(i);
      }
      Operation *sum_broadcast = create_broadcast_in_dim(
          loc, sum->getResult(0), broadcast_dims, scores_type, rewriter);

      OperationState div_state(loc, "stablehlo.divide");
      div_state.addOperands({exp_result->getResult(0), sum_broadcast->getResult(0)});
      div_state.addTypes(scores_type);
      Operation *attn_probs = rewriter.create(div_state);

      // Stable softmax may use greater precision than the value matmul.
      Value dot_lhs = attn_probs->getResult(0);
      if (scores_type.getElementType() != value_type.getElementType()) {
        auto converted_type = RankedTensorType::get(
            scores_type.getShape(), value_type.getElementType());
        OperationState cvt_state(loc, "stablehlo.convert");
        cvt_state.addOperands(dot_lhs);
        cvt_state.addTypes(converted_type);
        Operation *cvt = rewriter.create(cvt_state);
        dot_lhs = cvt->getResult(0);
      }

      Operation *result = create_dot_general_with_dims(loc, dot_lhs, v,
                                                        value_dot_dims,
                                                        op->getResultTypes(), rewriter);
      if (!result) return failure();
      rewriter.replaceOp(op, result->getResults());
      return success();
    }

    if (pattern == "attention" && inputs.size() == 3) {
      Location loc = op.getLoc();
      Value q = inputs[0];
      Value k = inputs[1];
      Value v = inputs[2];
      auto scale = get_backend_config_float(op, "zigrad.scale");
      auto reduction_dim =
          get_backend_config_int(op, "zigrad.reduction_dim");
      Attribute score_dot_dims = get_backend_config_attr(op, "zigrad.score_dot_dims");
      Attribute value_dot_dims = get_backend_config_attr(op, "zigrad.value_dot_dims");
      auto backend_config =
          op->getAttrOfType<DictionaryAttr>("backend_config");
      auto scores_shape_attr = backend_config
                                   ? dyn_cast_or_null<DenseI64ArrayAttr>(
                                         backend_config.get("zigrad.scores_shape"))
                                   : DenseI64ArrayAttr();
      auto query_type = dyn_cast<RankedTensorType>(q.getType());
      auto value_type = dyn_cast<RankedTensorType>(v.getType());
      if (failed(scale) || failed(reduction_dim) || !score_dot_dims ||
          !value_dot_dims || !scores_shape_attr || !query_type || !value_type)
        return failure();

      SmallVector<int64_t> scores_dims(scores_shape_attr.asArrayRef());
      if (*reduction_dim < 0 ||
          *reduction_dim >= static_cast<int64_t>(scores_dims.size()))
        return failure();

      Type scores_element_type = query_type.getElementType();
      if (!isa<FloatType>(scores_element_type)) return failure();
      auto scores_type =
          RankedTensorType::get(scores_dims, scores_element_type);

      Operation *raw_scores = create_dot_general_with_dims(
          loc, q, k, score_dot_dims, {scores_type}, rewriter);
      if (!raw_scores) return failure();

      auto scalar_type = RankedTensorType::get({}, scores_element_type);
      auto scale_attr = DenseElementsAttr::get(
          scalar_type, rewriter.getFloatAttr(scores_element_type, *scale));
      OperationState scale_state(loc, "stablehlo.constant");
      scale_state.addTypes(scalar_type);
      scale_state.addAttribute("value", scale_attr);
      Operation *scale_const = rewriter.create(scale_state);

      SmallVector<int64_t> empty_dims;
      Operation *scale_broadcast = create_broadcast_in_dim(
          loc, scale_const->getResult(0), empty_dims, scores_type, rewriter);

      OperationState mul_state(loc, "stablehlo.multiply");
      mul_state.addOperands({raw_scores->getResult(0), scale_broadcast->getResult(0)});
      mul_state.addTypes(scores_type);
      Operation *scaled = rewriter.create(mul_state);

      Operation *max_val = create_reduce_max(
          loc, scaled->getResult(0), *reduction_dim, rewriter);

      SmallVector<int64_t> broadcast_dims;
      for (int64_t i = 0; i < scores_type.getRank(); ++i) {
        if (i != *reduction_dim) broadcast_dims.push_back(i);
      }
      Operation *max_broadcast = create_broadcast_in_dim(
          loc, max_val->getResult(0), broadcast_dims, scores_type, rewriter);

      OperationState sub_state(loc, "stablehlo.subtract");
      sub_state.addOperands({scaled->getResult(0), max_broadcast->getResult(0)});
      sub_state.addTypes(scores_type);
      Operation *shifted = rewriter.create(sub_state);

      OperationState exp_state(loc, "stablehlo.exponential");
      exp_state.addOperands(shifted->getResult(0));
      exp_state.addTypes(scores_type);
      Operation *exp_result = rewriter.create(exp_state);

      Operation *sum = create_reduce_sum(loc, exp_result->getResult(0),
                                         *reduction_dim, rewriter);

      Operation *sum_broadcast = create_broadcast_in_dim(
          loc, sum->getResult(0), broadcast_dims, scores_type, rewriter);

      OperationState div_state(loc, "stablehlo.divide");
      div_state.addOperands({exp_result->getResult(0), sum_broadcast->getResult(0)});
      div_state.addTypes(scores_type);
      Operation *probs = rewriter.create(div_state);

      // Stable softmax may use greater precision than the value matmul.
      Value dot_lhs = probs->getResult(0);
      if (scores_element_type != value_type.getElementType()) {
        auto converted_type = RankedTensorType::get(
            scores_dims, value_type.getElementType());
        OperationState cvt_state(loc, "stablehlo.convert");
        cvt_state.addOperands(dot_lhs);
        cvt_state.addTypes(converted_type);
        Operation *cvt = rewriter.create(cvt_state);
        dot_lhs = cvt->getResult(0);
      }

      Operation *result = create_dot_general_with_dims(
          loc, dot_lhs, v, value_dot_dims, op->getResultTypes(), rewriter);
      if (!result) return failure();
      rewriter.replaceOp(op, result->getResults());
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

  void getDependentDialects(DialectRegistry &registry) const override {
    MlirDialectRegistry cReg = wrap(&registry);
    mlirDialectHandleInsertDialect(mlirGetDialectHandle__stablehlo__(), cReg);
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<KernelCallExpandPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

static FailureOr<Attribute> build_row_major_layout(PatternRewriter &rewriter,
                                                    RankedTensorType ranked) {
  const int64_t rank = ranked.getRank();
  llvm::SmallVector<int64_t> order(rank);
  for (int64_t i = 0; i < rank; ++i) {
    order[i] = rank - i - 1;
  }
  auto layout_ty = RankedTensorType::get({rank}, rewriter.getIndexType());
  return DenseIntElementsAttr::get(layout_ty, order);
}

static FailureOr<ArrayAttr> build_default_layouts(PatternRewriter &rewriter,
                                                    TypeRange types) {
  SmallVector<Attribute> layouts;
  layouts.reserve(types.size());

  for (Type type : types) {
    auto ranked = dyn_cast<RankedTensorType>(type);
    if (!ranked) return failure();
    layouts.push_back(*build_row_major_layout(rewriter, ranked));
  }

  return rewriter.getArrayAttr(layouts);
}

struct KernelCallToStablehloCustomCallPattern final : OpRewritePattern<KernelCallOp> {
  using OpRewritePattern<KernelCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(KernelCallOp op, PatternRewriter &rewriter) const override {
    auto operand_layouts = build_default_layouts(rewriter, op.getInputs().getTypes());
    if (failed(operand_layouts)) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor operands");
    }

    auto result_layouts = build_default_layouts(rewriter, op->getResultTypes());
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

  void getDependentDialects(DialectRegistry &registry) const override {
    MlirDialectRegistry cReg = wrap(&registry);
    mlirDialectHandleInsertDialect(mlirGetDialectHandle__stablehlo__(), cReg);
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
}

} // namespace mlir::zigrad
