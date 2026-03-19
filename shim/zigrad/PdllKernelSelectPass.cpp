/// PDLL-based kernel selection pass (spike).
///
/// Registers a pass "zg-pdll-kernel-select" that runs PDLL-generated patterns
/// instead of the imperative C++ RewritePattern classes. This allows us to
/// validate PDLL end-to-end using mlir-opt with the existing test MLIR files.

#include "zigrad/PdllKernelSelectPass.h"
#include "zigrad/ZigradDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// StableHLO C API for dimension number checks in native constraints
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/c/StablehloAttributes.h"

// PDLL-generated code is emitted at file scope (static functions, no namespace).
// Pull in the MLIR types it references.
using namespace mlir;
using namespace llvm;

// PDLL-generated patterns (output of: mlir-pdll dot_add.pdll -x=cpp)
#include "zigrad/patterns/dot_add.pdll.cpp.inc"

// The include resolves to ${CMAKE_CURRENT_BINARY_DIR}/generated/zigrad/patterns/dot_add.pdll.cpp.inc

namespace mlir::zigrad {
namespace {

struct PdllKernelSelectPass
    : public PassWrapper<PdllKernelSelectPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PdllKernelSelectPass)

  StringRef getArgument() const override { return "zg-pdll-kernel-select"; }
  StringRef getDescription() const override {
    return "PDLL-based kernel selection (spike: DotAdd only).";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pdl::PDLDialect>();
    registry.insert<pdl_interp::PDLInterpDialect>();
    registry.insert<mlir::zigrad::ZigradDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGeneratedPDLLPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void registerPdllKernelSelectPass() {
  PassRegistration<PdllKernelSelectPass>();
}

} // namespace mlir::zigrad
