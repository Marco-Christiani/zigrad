include "mlir/IR/OpBase.td"

def Zigrad_Dialect : Dialect {
  let name = "zigrad";
  let cppNamespace = "::mlir::zigrad";
}

class Zigrad_Op<string mnemonic, list<Trait> traits = []>
    : Op<Zigrad_Dialect, mnemonic, traits>;

def Zigrad_KernelCallOp : Zigrad_Op<"kernel_call"> {
  let summary = "Kernel dispatch carrier operation";
  let description = [{
    Carrier operation used before legalization to StableHLO custom_call.
    Dispatch identity is carried through attributes and preserved by rewrite.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    I32Attr:$api_version,
    StrAttr:$call_target_name,
    BoolAttr:$has_side_effect,
    AnyAttr:$backend_config
  );

  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, $outputs)
  }];
}
