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

// ============================================================================
// Expand pass: revert kernel_call ops back to original StableHLO ops.
//
// Used when a provider returns Unsupported during materialization: the
// kernel_call is expanded back to the StableHLO pattern it was created from
// (dot_add -> add(dot(a,b), c), etc.) so the backend can handle it natively.
// ============================================================================

static StringRef get_backend_config_string(Operation *op, StringRef key) {
  auto bc = op->getAttrOfType<DictionaryAttr>("backend_config");
  if (!bc) return {};
  auto attr = bc.getAs<StringAttr>(key);
  if (!attr) return {};
  return attr.getValue();
}

/// Extract an i32 integer attribute from backend_config. Returns 0 if missing.
static int32_t get_backend_config_int(Operation *op, StringRef key) {
  auto bc = op->getAttrOfType<DictionaryAttr>("backend_config");
  if (!bc) return 0;
  auto attr = bc.getAs<IntegerAttr>(key);
  if (!attr) return 0;
  return static_cast<int32_t>(attr.getInt());
}

/// Extract an f32 float attribute from backend_config. Returns 0.0 if missing.
static float get_backend_config_float(Operation *op, StringRef key) {
  auto bc = op->getAttrOfType<DictionaryAttr>("backend_config");
  if (!bc) return 0.0f;
  auto attr = bc.getAs<FloatAttr>(key);
  if (!attr) return 0.0f;
  return static_cast<float>(attr.getValueAsDouble());
}

/// Extract a raw Attribute from backend_config. Returns nullptr if missing.
static Attribute get_backend_config_attr(Operation *op, StringRef key) {
  auto bc = op->getAttrOfType<DictionaryAttr>("backend_config");
  if (!bc) return {};
  return bc.get(key);
}

/// Build a stablehlo.reduce with an add combiner (reduce_sum) over a single
/// dimension. The init value is a scalar zero of the element type.
static Operation *create_reduce_sum(Location loc, Value input, int64_t dim,
                                     PatternRewriter &rewriter) {
  auto input_type = cast<RankedTensorType>(input.getType());
  Type element_type = input_type.getElementType();

  // Build zero init value.
  auto zero_type = RankedTensorType::get({}, element_type);
  auto zero_attr = DenseElementsAttr::get(zero_type, rewriter.getZeroAttr(element_type));
  OperationState zero_state(loc, "stablehlo.constant");
  zero_state.addTypes(zero_type);
  zero_state.addAttribute("value", zero_attr);
  Operation *zero_op = rewriter.create(zero_state);

  // Compute result shape (input shape with dim removed).
  SmallVector<int64_t> result_shape;
  for (int64_t i = 0; i < input_type.getRank(); ++i) {
    if (i != dim) result_shape.push_back(input_type.getDimSize(i));
  }
  auto result_type = RankedTensorType::get(result_shape, element_type);

  OperationState state(loc, "stablehlo.reduce");
  state.addOperands({input, zero_op->getResult(0)});
  state.addTypes(result_type);
  state.addAttribute("dimensions", rewriter.getDenseI64ArrayAttr({dim}));

  // Build the combiner body region.
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

/// Build a stablehlo.broadcast_in_dim to broadcast `input` into `result_type`
/// along the given `broadcast_dimensions`.
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

/// Build a stablehlo.reduce with a maximum combiner (reduce_max) over a single
/// dimension. The init value is negative infinity of the element type.
static Operation *create_reduce_max(Location loc, Value input, int64_t dim,
                                     PatternRewriter &rewriter) {
  auto input_type = cast<RankedTensorType>(input.getType());
  Type element_type = input_type.getElementType();

  // Build -inf init value.
  auto scalar_type = RankedTensorType::get({}, element_type);
  auto neg_inf = APFloat::getInf(
      cast<FloatType>(element_type).getFloatSemantics(), /*Negative=*/true);
  auto init_attr = DenseElementsAttr::get(scalar_type,
                                            rewriter.getFloatAttr(element_type, neg_inf));
  OperationState init_state(loc, "stablehlo.constant");
  init_state.addTypes(scalar_type);
  init_state.addAttribute("value", init_attr);
  Operation *init_op = rewriter.create(init_state);

  // Compute result shape (input shape with dim removed).
  SmallVector<int64_t> result_shape;
  for (int64_t i = 0; i < input_type.getRank(); ++i) {
    if (i != dim) result_shape.push_back(input_type.getDimSize(i));
  }
  auto result_type = RankedTensorType::get(result_shape, element_type);

  OperationState state(loc, "stablehlo.reduce");
  state.addOperands({input, init_op->getResult(0)});
  state.addTypes(result_type);
  state.addAttribute("dimensions", rewriter.getDenseI64ArrayAttr({dim}));

  // Build the combiner body region.
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

/// Build a stablehlo.dot_general op with explicit dimension numbers attribute.
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

    if (pattern == "rms_norm" && inputs.size() == 1) {
      // Expand: rms_norm(x) -> multiply(x, broadcast(rsqrt(add(multiply(reduce_sum(multiply(x,x)), scale), eps))))
      // Weight multiply stays outside the kernel boundary.
      Location loc = op.getLoc();
      Value x = inputs[0];
      int32_t normalized_size = get_backend_config_int(op, "zigrad.normalized_size");
      if (normalized_size <= 0) return failure();

      auto x_type = cast<RankedTensorType>(x.getType());
      Type elem = x_type.getElementType();
      int64_t last_dim = x_type.getRank() - 1;

      // x_sq = multiply(x, x)
      OperationState sq_state(loc, "stablehlo.multiply");
      sq_state.addOperands({x, x});
      sq_state.addTypes(x_type);
      Operation *x_sq = rewriter.create(sq_state);

      // sum = reduce_sum(x_sq, last_dim)
      Operation *sum = create_reduce_sum(loc, x_sq->getResult(0), last_dim, rewriter);

      // scale = constant(1.0 / normalized_size) broadcast to sum shape
      auto sum_type = cast<RankedTensorType>(sum->getResult(0).getType());
      auto scalar_type = RankedTensorType::get({}, elem);
      float scale_val = 1.0f / static_cast<float>(normalized_size);
      auto scale_attr = DenseElementsAttr::get(scalar_type, rewriter.getFloatAttr(elem, scale_val));
      OperationState scale_state(loc, "stablehlo.constant");
      scale_state.addTypes(scalar_type);
      scale_state.addAttribute("value", scale_attr);
      Operation *scale_const = rewriter.create(scale_state);

      // Broadcast scale to sum shape.
      SmallVector<int64_t> empty_dims;
      Operation *scale_broadcast = create_broadcast_in_dim(
          loc, scale_const->getResult(0), empty_dims, sum_type, rewriter);

      // mean = multiply(sum, scale_broadcast)
      OperationState mean_state(loc, "stablehlo.multiply");
      mean_state.addOperands({sum->getResult(0), scale_broadcast->getResult(0)});
      mean_state.addTypes(sum_type);
      Operation *mean = rewriter.create(mean_state);

      // eps = constant(1e-5) broadcast to sum shape
      auto eps_attr = DenseElementsAttr::get(scalar_type, rewriter.getFloatAttr(elem, 1.0e-5));
      OperationState eps_state(loc, "stablehlo.constant");
      eps_state.addTypes(scalar_type);
      eps_state.addAttribute("value", eps_attr);
      Operation *eps_const = rewriter.create(eps_state);
      Operation *eps_broadcast = create_broadcast_in_dim(
          loc, eps_const->getResult(0), empty_dims, sum_type, rewriter);

      // denom = add(mean, eps_broadcast)
      OperationState denom_state(loc, "stablehlo.add");
      denom_state.addOperands({mean->getResult(0), eps_broadcast->getResult(0)});
      denom_state.addTypes(sum_type);
      Operation *denom = rewriter.create(denom_state);

      // inv = rsqrt(denom)
      OperationState rsqrt_state(loc, "stablehlo.rsqrt");
      rsqrt_state.addOperands(denom->getResult(0));
      rsqrt_state.addTypes(sum_type);
      Operation *inv = rewriter.create(rsqrt_state);

      // broadcast inv to x shape
      SmallVector<int64_t> inv_broadcast_dims;
      for (int64_t i = 0; i < x_type.getRank(); ++i) {
        if (i != last_dim) inv_broadcast_dims.push_back(i);
      }
      Operation *inv_broadcast = create_broadcast_in_dim(
          loc, inv->getResult(0), inv_broadcast_dims, x_type, rewriter);

      // result = multiply(x, inv_broadcast) = normed
      OperationState normed_state(loc, "stablehlo.multiply");
      normed_state.addOperands({x, inv_broadcast->getResult(0)});
      normed_state.addTypes(op->getResultTypes());
      Operation *normed = rewriter.create(normed_state);
      rewriter.replaceOp(op, normed->getResults());
      return success();
    }

    if (pattern == "softmax_matmul" && inputs.size() == 2) {
      // Expand: softmax_matmul(scores, V) -> dot_general(div(exp(scores), broadcast(reduce_sum(exp(scores)))), V)
      Location loc = op.getLoc();
      Value scores = inputs[0];
      Value v = inputs[1];
      int32_t reduction_dim = get_backend_config_int(op, "zigrad.reduction_dim");

      auto scores_type = cast<RankedTensorType>(scores.getType());

      // exp_result = exponential(scores)
      OperationState exp_state(loc, "stablehlo.exponential");
      exp_state.addOperands(scores);
      exp_state.addTypes(scores_type);
      Operation *exp_result = rewriter.create(exp_state);

      // sum = reduce_sum(exp_result, reduction_dim)
      Operation *sum = create_reduce_sum(loc, exp_result->getResult(0),
                                          static_cast<int64_t>(reduction_dim), rewriter);

      // broadcast sum back to scores shape
      // broadcast_dims = all dims except reduction_dim
      SmallVector<int64_t> broadcast_dims;
      for (int64_t i = 0; i < scores_type.getRank(); ++i) {
        if (i != static_cast<int64_t>(reduction_dim))
          broadcast_dims.push_back(i);
      }
      Operation *sum_broadcast = create_broadcast_in_dim(
          loc, sum->getResult(0), broadcast_dims, scores_type, rewriter);

      // attn_probs = divide(exp_result, sum_broadcast)
      OperationState div_state(loc, "stablehlo.divide");
      div_state.addOperands({exp_result->getResult(0), sum_broadcast->getResult(0)});
      div_state.addTypes(scores_type);
      Operation *attn_probs = rewriter.create(div_state);

      // If scores dtype != V dtype (e.g. softmax in f32, V in bf16),
      // insert a convert to match V's element type before the dot_general.
      Value dot_lhs = attn_probs->getResult(0);
      auto v_type = cast<RankedTensorType>(v.getType());
      if (scores_type.getElementType() != v_type.getElementType()) {
        auto converted_type = RankedTensorType::get(
            scores_type.getShape(), v_type.getElementType());
        OperationState cvt_state(loc, "stablehlo.convert");
        cvt_state.addOperands(dot_lhs);
        cvt_state.addTypes(converted_type);
        Operation *cvt = rewriter.create(cvt_state);
        dot_lhs = cvt->getResult(0);
      }

      // result = dot_general(attn_probs, V)
      Operation *result = create_dot_general(loc, dot_lhs, v,
                                              op->getResultTypes(), rewriter);
      if (!result) return failure();
      rewriter.replaceOp(op, result->getResults());
      return success();
    }

    if (pattern == "attention" && inputs.size() == 3) {
      // Expand: attention(Q, K, V) -> dot_general(softmax(scale * dot_general(Q, K)), V)
      // with numerically stable softmax (max-shift).
      Location loc = op.getLoc();
      Value q = inputs[0];
      Value k = inputs[1];
      Value v = inputs[2];
      float scale_value = get_backend_config_float(op, "zigrad.scale");
      int32_t reduction_dim = get_backend_config_int(op, "zigrad.reduction_dim");
      Attribute score_dot_dims = get_backend_config_attr(op, "zigrad.score_dot_dims");
      Attribute value_dot_dims = get_backend_config_attr(op, "zigrad.value_dot_dims");
      auto scores_shape_attr = op->getAttrOfType<DenseI64ArrayAttr>(
          StringRef("backend_config"))
          ? DenseI64ArrayAttr()
          : DenseI64ArrayAttr();
      // Extract scores_shape from backend_config dict.
      {
        auto bc = op->getAttrOfType<DictionaryAttr>("backend_config");
        if (bc) scores_shape_attr = dyn_cast_or_null<DenseI64ArrayAttr>(bc.get("zigrad.scores_shape"));
      }
      if (!score_dot_dims || !value_dot_dims || !scores_shape_attr)
        return failure();

      // Determine scores type from stored shape.
      auto q_type = cast<RankedTensorType>(q.getType());
      Type scores_elem = q_type.getElementType();
      SmallVector<int64_t> scores_dims(scores_shape_attr.asArrayRef());
      auto scores_type = RankedTensorType::get(scores_dims, scores_elem);

      // raw_scores = dot_general(Q, K) using stored score_dot_dims
      Operation *raw_scores = create_dot_general_with_dims(
          loc, q, k, score_dot_dims, {scores_type}, rewriter);
      if (!raw_scores) return failure();

      // scaled = multiply(raw_scores, broadcast(constant(scale)))
      auto scalar_type = RankedTensorType::get({}, scores_elem);
      auto scale_attr = DenseElementsAttr::get(
          scalar_type, rewriter.getFloatAttr(scores_elem, scale_value));
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

      // max = reduce_max(scaled, reduction_dim)
      Operation *max_val = create_reduce_max(loc, scaled->getResult(0),
                                              static_cast<int64_t>(reduction_dim), rewriter);

      // broadcast max back to scores shape
      SmallVector<int64_t> broadcast_dims;
      for (int64_t i = 0; i < scores_type.getRank(); ++i) {
        if (i != static_cast<int64_t>(reduction_dim))
          broadcast_dims.push_back(i);
      }
      Operation *max_broadcast = create_broadcast_in_dim(
          loc, max_val->getResult(0), broadcast_dims, scores_type, rewriter);

      // shifted = subtract(scaled, max_broadcast)
      OperationState sub_state(loc, "stablehlo.subtract");
      sub_state.addOperands({scaled->getResult(0), max_broadcast->getResult(0)});
      sub_state.addTypes(scores_type);
      Operation *shifted = rewriter.create(sub_state);

      // exp = exponential(shifted)
      OperationState exp_state(loc, "stablehlo.exponential");
      exp_state.addOperands(shifted->getResult(0));
      exp_state.addTypes(scores_type);
      Operation *exp_result = rewriter.create(exp_state);

      // sum = reduce_sum(exp, reduction_dim)
      Operation *sum = create_reduce_sum(loc, exp_result->getResult(0),
                                          static_cast<int64_t>(reduction_dim), rewriter);

      // broadcast sum back to scores shape
      Operation *sum_broadcast = create_broadcast_in_dim(
          loc, sum->getResult(0), broadcast_dims, scores_type, rewriter);

      // probs = divide(exp, sum_broadcast)
      OperationState div_state(loc, "stablehlo.divide");
      div_state.addOperands({exp_result->getResult(0), sum_broadcast->getResult(0)});
      div_state.addTypes(scores_type);
      Operation *probs = rewriter.create(div_state);

      // Optional convert if probs dtype != V dtype.
      Value dot_lhs = probs->getResult(0);
      auto v_type = cast<RankedTensorType>(v.getType());
      if (scores_elem != v_type.getElementType()) {
        auto converted_type = RankedTensorType::get(
            scores_dims, v_type.getElementType());
        OperationState cvt_state(loc, "stablehlo.convert");
        cvt_state.addOperands(dot_lhs);
        cvt_state.addTypes(converted_type);
        Operation *cvt = rewriter.create(cvt_state);
        dot_lhs = cvt->getResult(0);
      }

      // result = dot_general(probs, V) using stored value_dot_dims
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

// ============================================================================
// Legalize pass: kernel_call -> stablehlo.custom_call
// ============================================================================

/// Build a row-major layout attribute for a single ranked tensor type.
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

/// Build default (row-major) layout attributes for a range of types.
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
