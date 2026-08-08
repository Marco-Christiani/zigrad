#include "mlir/CAPI/IR.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "stablehlo/integrations/c/StablehloDialect.h"

#include "zigrad/ZigradDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  registry.insert<mlir::zigrad::ZigradDialect>();

  MlirDialectRegistry c_registry = wrap(&registry);
  mlirDialectHandleInsertDialect(mlirGetDialectHandle__stablehlo__(),
                                 c_registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry)) ? 1 : 0;
}
