//! Native PJRT executable loading.

const Executor = @import("../execution.zig");
const compiler = @import("compiler.zig");
const execution = @import("execution.zig");

const LoaderInterface = Executor.Loader(compiler.Artifact);

/// Loader for one PJRT execution context.
pub const Loader = struct {
    interface: LoaderInterface,
    execution: *execution.Execution,

    const vtable: LoaderInterface.VTable = .{
        .load = load,
    };

    pub fn init(execution_instance: *execution.Execution) Loader {
        return .{
            .interface = .{
                .executor = &execution_instance.interface,
                .vtable = &vtable,
            },
            .execution = execution_instance,
        };
    }

    fn load(
        interface: *LoaderInterface,
        artifact: *compiler.Artifact,
    ) Executor.Error!*anyopaque {
        const self: *Loader = @fieldParentPtr("interface", interface);
        if (artifact.client != self.execution.client) return error.InvalidArgument;

        const handle = try self.execution.adopt_handle(artifact.loaded);
        artifact.* = undefined;
        return handle;
    }
};
