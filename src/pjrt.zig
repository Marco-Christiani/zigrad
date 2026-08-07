//! Public surface for the optional PJRT integration.

const client = @import("pjrt/client.zig");
const backend = @import("pjrt/backend.zig");
const compiler = @import("pjrt/compiler.zig");
const execution = @import("pjrt/execution.zig");
const loader = @import("pjrt/loader.zig");

pub const LoadedExecutable = client.LoadedExecutable;
pub const Buffer = client.Buffer;
pub const RawBuffer = client.RawBuffer;
pub const Device = client.Device;
pub const Event = client.Event;
pub const ExecuteResult = client.ExecuteResult;
pub const CompileOptions = client.CompileOptions;
pub const DispatchOptions = client.DispatchOptions;
pub const InitOptions = client.InitOptions;
pub const config = client.config;
pub const Client = client.Client;
pub const Execution = execution.Execution;
pub const Artifact = compiler.Artifact;
pub const Compiler = compiler.Compiler;
pub const Loader = loader.Loader;
pub const Backend = backend.Backend;

const dump = @import("pjrt/dump.zig");
pub const DumpConfig = dump.Config;
pub const DumpOptimized = dump.DumpOptimized;

test {
    @import("std").testing.refAllDecls(@This());
}
