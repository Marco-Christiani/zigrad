//! Optional IREE integration.

const build_options = @import("build_options");

const config = @import("iree/config.zig");
const backend = @import("iree/backend.zig");
const compiler = @import("iree/compiler.zig");
const execution = @import("iree/execution.zig");
const loader = @import("iree/loader.zig");
const runtime = @import("iree/runtime.zig");

pub const Config = config.Config;
pub const CompilerConfig = config.CompilerConfig;
pub const RuntimeConfig = config.RuntimeConfig;
pub const DeviceConstruction = runtime.DeviceConstruction;
pub const SourceEncoding = compiler.SourceEncoding;
pub const CompileOptions = compiler.CompileOptions;
pub const compile = compiler.compile;
pub const Artifact = compiler.Artifact;
pub const Compiler = compiler.Compiler;
pub const Execution = execution.Execution;
pub const Loader = loader.Loader;
pub const Backend = backend.Backend;
pub const ElementType = runtime.ElementType;
pub const Bytecode = runtime.Bytecode;
pub const Buffer = runtime.Buffer;
pub const Invocation = runtime.Invocation;
pub const Executable = runtime.Executable;
pub const Runtime = runtime.Runtime;

/// Default PR to IREE composition, available with the MLIR lowering integration.
pub const pipeline = if (build_options.has_mlir) @import("iree/pipeline.zig") else struct {};

test {
    @import("std").testing.refAllDecls(@This());
}

test "public IREE surface excludes runtime ABI declarations" {
    const testing = @import("std").testing;
    try testing.expect(!@hasDecl(@This(), "Device"));
    try testing.expect(!@hasDecl(@This(), "abi"));
    try testing.expect(!@hasDecl(@This(), "types"));
}
