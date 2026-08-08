#include "zigrad/ZigradDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::zigrad;

ZigradDialect::ZigradDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<ZigradDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "zigrad/ZigradOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "zigrad/ZigradOps.cpp.inc"
