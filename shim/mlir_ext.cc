#include "mlir-c/IR.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Transforms.h"

#include "mlir/CAPI/IR.h"

#include "zigrad/ZigradDialect.h"
#include "zigrad/ZigradKernelLegalizePass.h"

// dialect registration header, should come from our sdk derivation
#include "stablehlo/integrations/c/StablehloDialect.h"
#include "stablehlo/integrations/c/StablehloPasses.h"

extern "C" void zg_register_dialects(MlirContext ctx) {
  // func
  MlirDialectHandle func = mlirGetDialectHandle__func__();
  mlirDialectHandleRegisterDialect(func, ctx);
  mlirDialectHandleLoadDialect(func, ctx);

  // stablehlo
  MlirDialectHandle sh = mlirGetDialectHandle__stablehlo__();
  mlirDialectHandleRegisterDialect(sh, ctx);
  mlirDialectHandleLoadDialect(sh, ctx);

  // zigrad
  unwrap(ctx)->getOrLoadDialect<mlir::zigrad::ZigradDialect>();
}

extern "C" void zg_register_passes() {
  mlirRegisterAllPasses();
  mlirRegisterAllStablehloPasses();
  mlir::zigrad::registerZigradKernelLegalizePasses();
}
