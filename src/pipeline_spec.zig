// src/pipeline/spec.zig
//
// Pipeline Specification (PS) proposal.
//
// What this models (and what it intentionally does NOT model):
// - PS is an end-to-end selection that binds together:
//     (1) a primary toolchain (whole-program IM -> EA)
//     (2) a runtime (EA execution environment)
//     (3) a compile mode (JIT vs cache vs AOT), which is cross-cutting
//     (4) optional kernelization providers (region-level PR -> KA), used *within* the same
//         whole-program compilation unit by inserting explicit KA call boundaries in IM.
//
// This keeps the architecture "friendly" to multiple backends without trying to be fully agnostic:
// - PR remains Zigrad-owned semantics + transforms.
// - IM realization is per toolchain profile (StableHLO/MLIR is the default profile today).
// - Kernelization is orthogonal: TVM/Mirage/etc can be used to generate KAs for selected regions
//   while the primary toolchain remains XLA/IREE/etc.
//
// Example usage:
//
//   const ps = PipelineSpec{
//       .mode = .jit,
//       .im_profile = .stablehlo_mlir_text,
//       .primary = .{ .xla = .{ .opt_level = 3 } },
//       .runtime = .{ .pjrt = .{} },
//       .kernelization = .{
//           .enabled = true,
//           .selection = .explicit_only, // only regions annotated by the user/compiler
//           .providers = &.{
//               .{ .tvm = .{ .budget = .small } },
//           },
//       },
//   };
//   try ps.validate(); // catches clear incompatibilities early
//
// Advantages:
// - JIT vs AOT is configured once and validated against both toolchain and runtime.
// - Toolchain remains "whole program"; kernelization remains "region level".
// - Adding a new toolchain does not require changing PR; add a new IM realizer + toolchain adapter.
// - Adding a new kernel provider does not require changing toolchains; it plugs into kernelization.

const std = @import("std");

pub const CompileMode = enum {
    /// Compile to a runtime-bound executable handle (contextual EA).
    jit,

    /// Compile and emit a PJRT-serializable cache artifact (still JIT, not “true AOT”).
    ///
    /// For PJRT, this maps to `PJRT_Executable_Serialize` / `PJRT_Executable_DeserializeAndLoad`.
    jit_cache,

    /// “True AOT” (portable EA). Not necessarily supported by all toolchains.
    ///
    /// For XLA, this is the `xla::Compiler::CompileAheadOfTime` style API (not PJRT serialize).
    aot,
};

pub const ImProfile = enum {
    // Default profile for MLIR-based toolchains.
    // (We can add more profiles later without changing the PS structure.)
    stablehlo_mlir_text,
    stablehlo_mlir_bytecode,
};

pub const PrimaryToolchainKind = enum {
    xla,
    iree,
    custom,
};

pub const RuntimeKind = enum {
    pjrt,
    iree,
    embedded,
    custom,
};

pub const KernelProviderKind = enum {
    tvm, // schedule search / kernel generation
    mirage, // multi-level search / superoptimization
    vendor_dispatch, // explicit library/kernel selection
    custom,
};

pub const KernelSelection = enum {
    // Keep this conservative initially: regions must be explicitly marked.
    explicit_only,
};

pub const PipelineSpec = struct {
    mode: CompileMode,

    // The IM profile the primary toolchain consumes.
    // Today StableHLO/MLIR is the default profile; other profiles can be added as needed.
    im_profile: ImProfile = .stablehlo_mlir_bytecode,

    // Whole-program compiler toolchain (IM -> EA).
    primary: PrimaryToolchainSpec,

    // Execution environment (EA execution).
    runtime: RuntimeSpec,

    // Optional: region-level kernelization (PR subgraph -> KA -> IM boundary insertion).
    kernelization: KernelizationSpec = .{},

    pub const ValidateError = error{
        // Cross-cutting incompatibilities:
        EmbeddedCannotJit,
        CompileModeUnsupported,

        // Toolchain/runtime pairing is invalid:
        ToolchainRuntimeMismatch,

        // IM profile not accepted by toolchain:
        ImProfileNotAccepted,

        // Kernelization requested but not supported by this pipeline:
        KernelizationUnsupported,
        KernelizationEnabledNoProviders,
    };

    pub fn validate(self: *const PipelineSpec) ValidateError!void {
        // Mode vs runtime: embedded is assumed AOT-only by default.
        // (If you later build an embedded runtime that can JIT, relax this rule.)
        if ((self.mode == .jit or self.mode == .jit_cache) and self.runtime.kind() == .embedded)
            return error.EmbeddedCannotJit;

        // Toolchain/runtime pairing: keep it strict for now to avoid ambiguous execution semantics.
        // This can be relaxed later if you intentionally support cross-pair execution.
        if (!isCompatiblePair(self.primary.kind(), self.runtime.kind()))
            return error.ToolchainRuntimeMismatch;

        // IM profile acceptance: explicitly declare what the toolchain consumes.
        if (!self.primary.acceptsIm(self.im_profile))
            return error.ImProfileNotAccepted;

        // Compile mode support: some toolchains/runtimes may only support a subset of modes.
        if (!self.primary.supportsCompileMode(self.mode) or !self.runtime.supportsCompileMode(self.mode))
            return error.CompileModeUnsupported;

        // Kernelization support: this is only valid if the pipeline can represent KA call boundaries.
        if (self.kernelization.enabled) {
            if (self.kernelization.providers.len == 0)
                return error.KernelizationEnabledNoProviders;

            // For now we treat "KA call boundaries exist" as a pipeline property.
            // If a given toolchain/runtime cannot support KA invocation, disable kernelization for that PS.
            if (!self.primary.supportsKernelCalls() or !self.runtime.supportsKernelCalls())
                return error.KernelizationUnsupported;
        }
    }
};

pub fn isCompatiblePair(tc: PrimaryToolchainKind, rt: RuntimeKind) bool {
    return switch (tc) {
        .xla => rt == .pjrt or rt == .custom,
        .iree => rt == .iree or rt == .custom,
        .custom => true,
    };
}

pub const PrimaryToolchainSpec = union(PrimaryToolchainKind) {
    xla: XlaToolchainConfig,
    iree: IreeToolchainConfig,
    custom: CustomToolchainConfig,

    pub fn kind(self: *const PrimaryToolchainSpec) PrimaryToolchainKind {
        return @as(PrimaryToolchainKind, self.*);
    }

    pub fn supportsCompileMode(self: *const PrimaryToolchainSpec, mode: CompileMode) bool {
        return switch (self.*) {
            // Today: XLA integration is via PJRT (JIT). PJRT executable serialization is a cache artifact.
            // “True AOT” is a separate future toolchain shape (see `.internal/2026-01-16-00_XLA_AOT_NOTE.md`).
            .xla => mode == .jit or mode == .jit_cache,

            // Placeholder: IREE can support both JIT-like and AOT-like flows depending on configuration.
            // Keep permissive until we implement the actual adapters.
            .iree => true,

            .custom => true,
        };
    }

    pub fn acceptsIm(self: *const PrimaryToolchainSpec, im: ImProfile) bool {
        return switch (self.*) {
            .xla => switch (im) {
                .stablehlo_mlir_text, .stablehlo_mlir_bytecode => true,
            },
            .iree => switch (im) {
                .stablehlo_mlir_text, .stablehlo_mlir_bytecode => true,
            },
            .custom => true, // custom toolchains define their own acceptance
        };
    }

    pub fn supportsKernelCalls(self: *const PrimaryToolchainSpec) bool {
        // Kernel calls are represented as explicit call boundaries in IM.
        // XLA (via custom calls) and IREE (via calls/external dispatch) can support this pattern.
        // Keep the rule permissive for custom toolchains.
        return switch (self.*) {
            .xla => true,
            .iree => true,
            .custom => true,
        };
    }
};

pub const RuntimeSpec = union(RuntimeKind) {
    pjrt: PjrtRuntimeConfig,
    iree: IreeRuntimeConfig,
    embedded: EmbeddedRuntimeConfig,
    custom: CustomRuntimeConfig,

    pub fn kind(self: *const RuntimeSpec) RuntimeKind {
        return @as(RuntimeKind, self.*);
    }

    pub fn supportsCompileMode(self: *const RuntimeSpec, mode: CompileMode) bool {
        return switch (self.*) {
            // PJRT runtime supports JIT compilation (via client) and deserialization for cache reuse.
            .pjrt => mode == .jit or mode == .jit_cache,

            // Placeholder: IREE can do both depending on deployment; keep permissive for now.
            .iree => true,

            // Default assumption: embedded deployments are AOT-only unless proven otherwise.
            .embedded => mode == .aot,

            .custom => true,
        };
    }

    pub fn supportsKernelCalls(self: *const RuntimeSpec) bool {
        // Kernel calls are invoked by the EA/runtime boundary; support depends on the runtime.
        // Keep permissive for now; if a runtime cannot support KA invocation, set this false.
        return switch (self.*) {
            .pjrt => true,
            .iree => true,
            .embedded => true, // embedded can still call kernels if they are linked/packaged appropriately
            .custom => true,
        };
    }
};

pub const KernelizationSpec = struct {
    enabled: bool = false,

    // Keep conservative: kernelization applies only where explicitly marked.
    selection: KernelSelection = .explicit_only,

    // One or more providers can be enabled. Provider choice is policy, not semantics.
    providers: []const KernelProviderSpec = &.{},
};

pub const KernelProviderSpec = union(KernelProviderKind) {
    tvm: TvmKernelProviderConfig,
    mirage: MirageKernelProviderConfig,
    vendor_dispatch: VendorDispatchConfig,
    custom: CustomKernelProviderConfig,
};

// Toolchain configs: keep these small; detailed knobs belong in toolchain modules.
pub const XlaToolchainConfig = struct {
    opt_level: u8 = 3,
};

pub const IreeToolchainConfig = struct {};

pub const CustomToolchainConfig = struct {};

// Runtime configs: keep these small.
pub const PjrtRuntimeConfig = struct {
    /// Explicit plugin DSO path (CPU/GPU plugin).
    /// This is intentionally PS-level config because it selects a concrete runtime.
    plugin_path: []const u8,
};
pub const IreeRuntimeConfig = struct {};
pub const EmbeddedRuntimeConfig = struct {};
pub const CustomRuntimeConfig = struct {};

// Kernel provider configs: keep these small; detailed knobs belong in provider modules.
pub const TvmKernelProviderConfig = struct {
    // High-level budget class; provider-specific details stay inside the provider.
    budget: enum { tiny, small, medium, large } = .small,
};

pub const MirageKernelProviderConfig = struct {};

pub const VendorDispatchConfig = struct {};

pub const CustomKernelProviderConfig = struct {};

test "PipelineSpec validation basic (xla+pjrt jit)" {
    var ps = PipelineSpec{
        .mode = .jit,
        .im_profile = .stablehlo_mlir_bytecode,
        .primary = .{ .xla = .{} },
        .runtime = .{ .pjrt = .{ .plugin_path = "/tmp/does-not-matter-for-validate" } },
    };
    try ps.validate();
}

test "PipelineSpec rejects true AOT for xla+pjrt today" {
    var ps = PipelineSpec{
        .mode = .aot,
        .im_profile = .stablehlo_mlir_bytecode,
        .primary = .{ .xla = .{} },
        .runtime = .{ .pjrt = .{ .plugin_path = "/tmp/does-not-matter-for-validate" } },
    };
    try std.testing.expectError(error.CompileModeUnsupported, ps.validate());
}
