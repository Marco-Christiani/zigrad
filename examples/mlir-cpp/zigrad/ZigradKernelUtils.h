#pragma once

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/integrations/c/StablehloAttributes.h"

namespace mlir::zigrad {
namespace kernel_utils {

constexpr const char *kDispatchTargetName = "zigrad.kernel.dispatch";
constexpr int kTypedFfiApiVersion = 4;

inline bool is_dot_op(Operation *op) {
  if (!op) return false;
  const StringRef name = op->getName().getStringRef();
  return name == "stablehlo.dot_general" || name == "stablehlo.dot";
}

inline bool is_named_op(Operation *op, StringRef name) {
  return op != nullptr && op->getName().getStringRef() == name;
}

/// Creates a kernel carrier operation for a selected StableHLO region.
///
/// `provider` and `kernel_key` are embedded in the backend_config dictionary.
///  Pattern-specific attributes are appended from `extra_config`.
inline Operation *create_kernel_call(Operation *anchor,
                                     ValueRange operands,
                                     TypeRange result_types,
                                     StringRef provider,
                                     StringRef kernel_key,
                                     StringRef pattern,
                                     ArrayRef<NamedAttribute> extra_config,
                                     PatternRewriter &rewriter) {
  NamedAttrList backend_fields;
  backend_fields.append("zigrad.kernel_key", rewriter.getStringAttr(kernel_key));
  backend_fields.append("zigrad.provider", rewriter.getStringAttr(provider));
  backend_fields.append("zigrad.pattern", rewriter.getStringAttr(pattern));
  for (auto &attr : extra_config)
    backend_fields.append(attr);

  OperationState state(anchor->getLoc(), "zigrad.kernel_call");
  state.addOperands(operands);
  state.addTypes(result_types);
  state.addAttribute("api_version", rewriter.getI32IntegerAttr(kTypedFfiApiVersion));
  state.addAttribute("call_target_name", rewriter.getStringAttr(kDispatchTargetName));
  state.addAttribute("has_side_effect", rewriter.getBoolAttr(false));
  state.addAttribute("backend_config", rewriter.getDictionaryAttr(backend_fields));

  return rewriter.create(state);
}

inline Operation *create_kernel_call(Operation *anchor,
                                     ValueRange operands,
                                     TypeRange result_types,
                                     StringRef provider,
                                     StringRef kernel_key,
                                     StringRef pattern,
                                     PatternRewriter &rewriter) {
  return create_kernel_call(anchor, operands, result_types, provider,
                            kernel_key, pattern, /*extra_config=*/{}, rewriter);
}

enum class RhsLayout {
  standard,
  transposed,
};

/// Checks the dimension numbers accepted by the Mirage matmul patterns.
///
/// 1. Equal rank for both operands
/// 2. Exactly one contracting dimension on the last LHS dimension
/// 3. The RHS contraction follows `rhs_layout`
/// 4. Batch dims are the leading prefix [0, 1, ..., rank-3]
inline bool has_matmul_dims(Operation *dot_op, RhsLayout rhs_layout) {
  if (!dot_op || dot_op->getNumOperands() < 2) return false;

  Attribute dims_attr = dot_op->getAttr("dot_dimension_numbers");
  if (!dims_attr) return false;

  MlirAttribute capi_attr = wrap(dims_attr);
  if (!stablehloAttributeIsADotDimensionNumbers(capi_attr)) return false;

  auto lhs_type = dyn_cast<RankedTensorType>(dot_op->getOperand(0).getType());
  auto rhs_type = dyn_cast<RankedTensorType>(dot_op->getOperand(1).getType());
  if (!lhs_type || !rhs_type) return false;

  int64_t lhs_rank = lhs_type.getRank();
  int64_t rhs_rank = rhs_type.getRank();

  if (lhs_rank != rhs_rank) return false;

  intptr_t n_lhs_contract =
      stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(capi_attr);
  intptr_t n_rhs_contract =
      stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(capi_attr);
  if (n_lhs_contract != 1 || n_rhs_contract != 1) return false;

  int64_t lhs_contract =
      stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(capi_attr, 0);
  int64_t rhs_contract =
      stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(capi_attr, 0);
  if (lhs_contract != lhs_rank - 1) return false;
  const int64_t expected_rhs_contract =
      rhs_layout == RhsLayout::standard ? rhs_rank - 2 : rhs_rank - 1;
  if (rhs_contract != expected_rhs_contract) return false;

  int64_t expected_batch = lhs_rank - 2;
  intptr_t n_lhs_batch =
      stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(capi_attr);
  intptr_t n_rhs_batch =
      stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(capi_attr);
  if (n_lhs_batch != expected_batch || n_rhs_batch != expected_batch)
    return false;

  for (int64_t i = 0; i < expected_batch; ++i) {
    if (stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(capi_attr, i) != i)
      return false;
    if (stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(capi_attr, i) != i)
      return false;
  }

  return true;
}

} // namespace kernel_utils
} // namespace mlir::zigrad
