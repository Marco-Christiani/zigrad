//! IREE VM bytecode loading.

const Executor = @import("../execution.zig");
const compiler = @import("compiler.zig");
const execution = @import("execution.zig");
const runtime = @import("runtime.zig");

const LoaderInterface = Executor.Loader(compiler.Artifact);

/// Loader for one IREE runtime and entry function.
pub const Loader = struct {
    interface: LoaderInterface,
    execution: *execution.Execution,
    entry_name: []const u8,

    const vtable: LoaderInterface.VTable = .{
        .load = load,
    };

    pub fn init(
        execution_instance: *execution.Execution,
        entry_name: []const u8,
    ) Loader {
        return .{
            .interface = .{
                .executor = &execution_instance.interface,
                .vtable = &vtable,
            },
            .execution = execution_instance,
            .entry_name = entry_name,
        };
    }

    fn load(
        interface: *LoaderInterface,
        artifact: *compiler.Artifact,
    ) Executor.Error!*anyopaque {
        const self: *Loader = @fieldParentPtr("interface", interface);
        var bytecode: runtime.Bytecode = .{
            .owned = .{
                .bytes = artifact.bytes,
                .allocator = artifact.allocator,
            },
        };
        const loaded = self.execution.runtime.load(
            artifact.allocator,
            &bytecode,
            self.entry_name,
        ) catch |err| return execution.map_error(err);
        artifact.* = undefined;
        return loaded._state;
    }
};
