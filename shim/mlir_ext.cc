#include <cstdio>

#include "mlir-c/IR.h"
#include "mlir-c/Dialect/Func.h"

// dialect registration header, should come from our sdk derivation
#include "stablehlo/integrations/c/StablehloDialect.h"

extern "C" void zg_register_dialects(MlirContext ctx) {
  // func
  MlirDialectHandle func = mlirGetDialectHandle__func__();
  mlirDialectHandleRegisterDialect(func, ctx);
  mlirDialectHandleLoadDialect(func, ctx);

  // stablehlo
  MlirDialectHandle sh = mlirGetDialectHandle__stablehlo__();
  mlirDialectHandleRegisterDialect(sh, ctx);
  mlirDialectHandleLoadDialect(sh, ctx);
}

