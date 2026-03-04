#pragma once

/// Reusable utilities for kernel provider passes.
///
/// Shared helpers for matching StableHLO patterns (is_dot_op, is_named_op) and
/// constructing zigrad.kernel_call carrier ops. Each provider pass (e.g.
/// MirageKernelSelectPass) uses these to emit kernel_calls without duplicating
/// the OperationState boilerplate.

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

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

} // namespace kernel_utils
} // namespace mlir::zigrad
