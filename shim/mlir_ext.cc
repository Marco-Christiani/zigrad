#include "mlir-c/IR.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Transforms.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "zigrad/ZigradDialect.h"
#include "zigrad/ZigradKernelLegalizePass.h"
#include "zigrad/mirage/MirageKernelSelectPass.h"
#include "zigrad/PdllKernelSelectPass.h"

// dialect registration header, should come from our sdk derivation
#include "stablehlo/integrations/c/StablehloDialect.h"
#include "stablehlo/integrations/c/StablehloPasses.h"

// ============================================================================
// C entry points
// ============================================================================

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
  mlir::zigrad::mirage::registerMirageKernelSelectPass();
  mlir::zigrad::registerPdllKernelSelectPass();
}

// ============================================================================
// mlir-opt plugin ABI hooks (for --load-dialect-plugin / --load-pass-plugin)
// ============================================================================

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "ZigradDialect", "0.1",
          [](::mlir::DialectRegistry *registry) {
            registry->insert<mlir::zigrad::ZigradDialect>();
            MlirDialectRegistry cReg = wrap(registry);
            mlirDialectHandleInsertDialect(
                mlirGetDialectHandle__stablehlo__(), cReg);
            mlirDialectHandleInsertDialect(mlirGetDialectHandle__func__(),
                                           cReg);
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "ZigradPasses", "0.1",
          []() {
            mlirRegisterAllStablehloPasses();
            mlir::zigrad::registerZigradKernelLegalizePasses();
            mlir::zigrad::mirage::registerMirageKernelSelectPass();
            mlir::zigrad::registerPdllKernelSelectPass();
          }};
}
