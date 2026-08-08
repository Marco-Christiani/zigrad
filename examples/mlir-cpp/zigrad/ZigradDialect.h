#pragma once

#include "llvm/ADT/StringRef.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::zigrad {

class ZigradDialect : public Dialect {
public:
  explicit ZigradDialect(MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "zigrad"; }
};

} // namespace mlir::zigrad

#define GET_OP_CLASSES
#include "zigrad/ZigradOps.h.inc"
