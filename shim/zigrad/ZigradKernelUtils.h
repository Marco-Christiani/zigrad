#pragma once

/// Reusable utilities for kernel provider passes.
///
/// Shared helpers for matching StableHLO patterns (is_dot_op, is_named_op) and
/// constructing zigrad.kernel_call carrier ops. Each provider pass (e.g.
/// MirageKernelSelectPass) uses these to emit kernel_calls without duplicating
/// the OperationState boilerplate.

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/integrations/c/StablehloAttributes.h"

namespace mlir::zigrad {
namespace kernel_utils {

constexpr const char *kDispatchTargetName = "zigrad.kernel.dispatch";
constexpr int kTypedFfiApiVersion = 4;

/// Check whether an operation is a dot-product source (dot_general or dot).
inline bool is_dot_op(Operation *op) {
  if (!op) return false;
  const StringRef name = op->getName().getStringRef();
  return name == "stablehlo.dot_general" || name == "stablehlo.dot";
}

/// Check whether an operation has a specific op name.
inline bool is_named_op(Operation *op, StringRef name) {
  return op != nullptr && op->getName().getStringRef() == name;
}

/// Build a `zigrad.kernel_call` operation from the given anchor location,
/// operands, result types, and kernel metadata.
///
/// `provider` and `kernel_key` are embedded in the backend_config dictionary.
/// Extra pattern-specific attributes (e.g. normalized_size, reduction_dim) are
/// appended from `extra_config`.
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

/// Convenience overload without extra config attributes.
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

/// Returns true iff a dot_general op has canonical matmul dimension numbers:
/// 1. Equal rank for both operands
/// 2. Exactly one contracting dim: LHS at rank-1, RHS at rank-2
/// 3. Batch dims are the leading prefix [0, 1, ..., rank-3]
inline bool has_canonical_matmul_dims(Operation *dot_op) {
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
  if (rhs_contract != rhs_rank - 2) return false;

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
