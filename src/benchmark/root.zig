//! Benchmarking harness for comparing kernel implementations across backends.
//!
//! Provides configurable matrix-multiply benchmarks with correctness checking,
//!  timing statistics, and adapters for TVM and XLA execution.
//! TODO: this is kind of legacy scratch work and should be formalized, extracted, or deleted.
pub const harness = @import("harness.zig");
pub const config = @import("config.zig");
pub const stats = @import("stats.zig");
pub const correctness = @import("correctness.zig");
pub const tvm_adapter = @import("tvm_adapter.zig");
pub const xla_adapter = @import("xla_adapter.zig");

pub const Harness = harness.Harness;
pub const BenchmarkConfig = config.BenchmarkConfig;
pub const Shape = config.Shape;
pub const Implementation = config.Implementation;

test {
    @import("std").testing.refAllDecls(@This());
}
