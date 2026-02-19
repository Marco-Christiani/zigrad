# v1 Transformation Plan

**Purpose:** Concrete plan for transforming the zigrad codebase from v0 (working prototype) to v1 (correct architecture). This is a rewrite informed by the prototype, not a refactoring of it.

**Date:** 2026-02-16

---

## Ground Rules

These rules govern all phases. They take precedence over phase-specific guidance.

### Do Not

- **Do not copy v0 code.** Read it, understand the algorithm, write it fresh. Copying carries structural assumptions.
- **Do not create files preemptively.** Create files when you write code that goes in them.
- **Do not create options structs for <3 fields.** Pass them as parameters.
- **Do not create enums with 2 variants.** Use a bool.
- **Do not create wrapper types for single values.** Use the value.
- **Do not split files under ~400 lines of logic.** Merge related small things.
- **Do not create utility modules.** If a helper is used in one place, it lives in that place.
- **Do not hide complexity behind abstraction.** Zigrad's philosophy is opt-in control. Every layer is accessible. Create layers where each layer uses the layer below it visibly.
- **Do not make decisions about future phases.** Each phase is designed when it begins, using what was learned from previous phases.

### Human-in-the-Loop Protocol

Before making any of the following decisions, STOP and present options to the human:
- Creating a new file or module
- Defining a new public type or interface
- Choosing between alternative representations
- Any cross-cutting concern (touches >2 modules)
- Any case where you are uncertain whether two things should be merged or separate
- Naming decisions for public-facing types

### Abstraction Layer Protocol

When inserting an abstraction layer (interface, trait, generic boundary), do NOT sketch code immediately. Instead:
1. Enter an architectural discussion with the human
2. Research the reference codebases and relevant external materials
3. Present concrete design alternatives with tradeoffs
4. Get human approval on the approach before writing any code

A wrong abstraction boundary propagates to every file that touches it. The cost of getting it right upfront is low; the cost of fixing it later is high.

### Foreign Type Containment

**No foreign types cross module boundaries.** This is the strongest form of the rule:
- Not C types from `@cImport`
- Not opaque handles that require foreign-API knowledge to use
- Not string-keyed function dispatch (e.g., `ffi_call("runtime.Foo", ...)`)
- Not foreign error codes or status types

If a module's caller needs to know anything about the foreign API to use it correctly, the boundary is in the wrong place. Integration modules must present Zig types with Zig semantics. All type coercion, string-based dispatch, error translation, and C++ object lifecycle are internal to the integration module.

---

## Relationship to v0

The v0 codebase is a reference, not a starting point. It demonstrates that the algorithms work (AD, StableHLO lowering, PJRT compilation, model training). It does NOT demonstrate correct architecture.

**How to use v0:**
- **Read it to understand behavior.** What does PJRT's compile API expect? What events does execute return? What donation flags exist? The v0 code answers these because it works.
- **Do not read it to understand structure.** The file layout, module boundaries, type locations, and import graph are all wrong.
- **Consider sweeping redesigns.** If v0 solves something in 200 lines but a fundamentally different approach would be cleaner in v1, prefer the redesign. The only thing that must be preserved is correctness.

**What to learn from v0 (behavior, not structure):**

| What to learn | Where to look in v0 | What to ignore |
|---|---|---|
| Op semantics: validation, type inference, AD rules | `src/pr/ops/*.zig` - `validate()`, `infer_output()`, `vjp_*` | The `lower()` methods, all MLIR imports |
| StableHLO mappings: how each PR op maps to StableHLO | `src/pr/ops/*.zig` `lower()` + `src/lower/` | File organization, co-location of lowering with semantics |
| PJRT API usage: compile, execute, buffer, event, device | `src/backend/` | Module structure, leaked types |
| What users need from the API | `examples/llama/` loss_fn + training loop | Raw PJRT buffer manipulation (the leak we're fixing) |
| Pass chain mechanics | `src/pipeline/` | The pjrt_types import in pass.zig |
| MLIR C bindings | `src/ffi/mlir/` | Nothing - self-contained and correct |
| PJRT C bindings | `src/ffi/pjrt/` | Nothing - self-contained and correct |

---

## Phase 1: PR Isolation

**Goal:** PR compiles with zero external dependencies beyond `std`.

Write the PR module fresh. Use v0 `src/pr/` as reference for op semantics.

**PR contains:**
- Op definitions: each op has `validate()`, `infer_output()`, and AD rules
- Program graph structure (nodes, edges, types)
- Context types: ValidateContext, InferContext, AdContext, FormatContext
- Region identification and queryable region access (e.g., iterate/extract regions matching annotation criteria)
- ZXPR pretty-print
- Annotations (semantic, steering, debug) as metadata on ops/regions

**PR does NOT contain:** Any `lower()` method, any MLIR import, LowerContext, any reference to StableHLO/PJRT/TVM or any backend.

**Decisions to ask the human about:**
- File organization within PR
- Region representation (new - v0 doesn't have it)
- Annotation representation

**Verification:**
- PR module compiles with only `std` imports
- Every op from v0 has validate + infer + AD in the new PR
- ZXPR pretty-print produces equivalent output to v0

---

## Phase 2: Lowering Module

**Goal:** A standalone module that maps PR ops to StableHLO. All MLIR imports live here.

Write the lowering module fresh. Use v0's `lower()` methods and `src/lower/` as reference.

**Lowering contains:**
- A dispatch mechanism mapping each PR op to StableHLO emission
- LowerContext (owns MLIR context, block, location, value map)
- The StableHLO lowering pass (consumes PR, produces IM)

**Lowering imports:** PR (to read ops) and MLIR bindings (to emit MLIR). This is the ONLY module that bridges PR and MLIR.

**Decisions to ask the human about:**
- Dispatch mechanism (switch vs table vs individual functions)
- File count (check actual line counts from v0 before proposing)

**Verification:**
- Lowering module compiles with PR + MLIR bindings as its only dependencies
- PR module still compiles without lowering (dependency is one-way)
- Produces equivalent StableHLO to v0 for representative ops

---

## Phase 3: Pass Infrastructure

**Goal:** The pass chain mechanism that threads everything together.

Write the pass infrastructure fresh. Use v0 `src/pipeline/` as reference.

**Pass infrastructure contains:**
- Pass trait: declares input format, output format, and a run function
- PassContext: allocator + error log + extensions
- Pass chain composition: validate adjacent pass compatibility, run in sequence

**Key design points:**
- Passes can operate on any format (PR, IM/MLIR, etc.) - the infrastructure only checks that adjacent passes declare compatible input/output formats.
- There is no wrapper type for "the thing flowing between passes." A PR is a PR, an IM is an IM. The pass format enum declares what a pass expects and produces.
- The pipeline can produce IM (via a lowering pass). Compilation (IM -> EA) stays in the backend.
- No backend-specific types in the pass infrastructure.
- A kernelization pass can insert KA boundary references (custom calls/sentinels) for kernels that don't exist yet. These are forward references - the actual KAs are compiled outside the pipeline and resolved at execution time via a KA registry. The pipeline runs straight through without splitting.

**Decisions to ask the human about:**
- PassContext extensions mechanism
- Error accumulation across pass boundaries (v0 loses error info here)
- Module naming and organization for bindings (see below)

**Verification:**
- Pass infrastructure compiles without backend or binding imports
- Can compose a chain of passes (including lowering) and run them
- Error context survives across pass boundaries

---

## Phase 4: PJRT Backend

**Goal:** Write the PJRT backend behind a Backend interface shaped by `docs/DESIGN.md` §4.1.

The Backend interface is design-driven: we know we need multiple backends, so we define the interface from the design document rather than extracting it from two implementations. But we do not over-engineer it for hypothetical backends - the interface is what DESIGN.md §4.1 describes and what PJRT needs, nothing more.

**Backend interface provides (per DESIGN.md):**
- `compile(IM, options) -> ExecutableArtifact`
- `execute(EA, inputs, exec_options) -> results`
- `get_devices() -> device list`
- `transfer(device, buffer, direction) -> transferred buffer`

**PJRT implementation:** Wraps the PJRT client behind the interface. All PJRT types are internal. The foreign type containment rule applies - no PJRT C types cross the backend boundary.

**Decisions to ask the human about:**
- Interface mechanism (tagged union, function pointers, comptime generic)
- Capability expression (compile-only vs execute-only vs both)
- Buffer types crossing the boundary
- Module naming for bindings vs integration layers

**Verification:**
- Backend interface compiles without any backend SDK imports
- PJRT implementation passes existing integration tests
- Pipeline can call backend.compile() and backend.execute() without knowing it's PJRT

---

## Phases 5-9: Scope Only

These phases are not designed yet. They will be designed when they begin, informed by what was learned in previous phases. Only the goal and dependencies are stated here.

**Phase 5: KernelProvider Interface.** Establish the extension point for kernel providers. Implement the KA registry that the backend consults at execution time to resolve KA boundary references. The kernelization pass (Phase 3) inserts forward references; the registry maps them to compiled KAs. Tuning integration is designed in this phase. Depends on Phases 3 and 4.

### Phase 5 Checkpoint (2026-02-19)

Current implementation status:
1. Kernelization pass now rewrites selected annotated regions into PR `custom_call` ops.
2. Temporary single-dispatch strategy is active using target `zigrad.kernel.dispatch`.
3. Dispatch metadata is carried in PR params and lowered into typed-FFI `backend_config` keys:
   - `zigrad.kernel_key`
   - `zigrad.provider`
4. Demo path has an end-to-end mode (`kernel-provider-demo`) that runs kernelize -> lower -> compile -> execute.
5. Policy is hard-fail when typed-FFI extension is unavailable.

Known blocker for full end-to-end completion:
1. PJRT typed-FFI handler registration for `zigrad.kernel.dispatch` is not wired yet.
2. Runtime currently fails with "No FFI handler registered ..." even though kernelization and lowering paths are active.

Next implementation target:
1. Register the typed-FFI dispatcher handler in PJRT integration and route calls through `KernelRegistry` lookup.

**Phase 6: TVM KernelProvider.** Implement TVM as a kernel provider behind the Phase 5 interface. The foreign type containment rule is critical here - v0's TVM integration is the primary example of what went wrong. Depends on Phase 5.

**Phase 7: High-Level Frontend.** A stable user-facing API that training examples (llama) are written against. This layer insulates examples from core API churn - when the core changes, only this layer adapts. Covers model definition (builder with Tensors), compilation, training step (forward + backward + parameter update), and data movement. The core APIs (pipeline, backend, passes) remain accessible for users who need direct control. Depends on Phases 3 and 4.

**Phase 8: IREE Backend.** Implement IREE as a second backend behind the Phase 4 interface. Validates the Backend interface design and the compile/execute capability separation (IREE genuinely separates these). Depends on Phase 4. Requires human intervention.

**Phase 9: Mirage KernelProvider.** Implement Mirage (the superoptimizer, not MPK runtime) as a second kernel provider behind the Phase 5 interface. Validates the KernelProvider interface design, particularly around tuning integration (Mirage's search is likely integral to compilation, unlike TVM where tuning is separable). Depends on Phase 5. Requires human intervention.

---

## Phase Dependencies

```
Phase 1: PR Isolation
   │
   ├─── Phase 2: Lowering Module (needs PR)
   │
   ├─── Phase 3: Pass Infrastructure (needs PR)
   │
   └─── [after 2+3] Phase 4: PJRT Backend (needs Pass Infra + Lowering)
                      │
                      ├─── Phase 5: KernelProvider Interface
                      │       │
                      │       ├─── Phase 6: TVM Provider
                      │       │
                      │       └─── Phase 9: Mirage Provider
                      │
                      ├─── Phase 7: High-Level Frontend
                      │
                      └─── Phase 8: IREE Backend
```

Phases 2 and 3 can run in parallel after Phase 1. Each phase produces a compilable, testable artifact. The human reviews and approves each phase before the next begins.

---

## Open Questions (to be resolved as phases progress)

- **Binding module naming and organization:** The current `src/ffi/` conflates several different things: `@cImport` wrappers for C APIs, C++ DSO integration with dlopen/object construction, and XLA's custom call FFI. These are different mechanisms with different containment requirements. The naming and organization is a human decision point. Note: "FFI" is established terminology in the XLA ecosystem for custom calls - we should not fight it in that context.
- **Tuning integration:** Tuning is a kernel provider concern. It is expensive, cacheable, and may require hardware access. The pipeline does not tune - it inserts KA boundary references that may or may not have tuning data behind them. The key design questions for Phase 5 are:
    - How does the user invoke tuning? (CLI command, API call, pipeline mode, or other)
    - How does a kernel provider access region information for tuning? PR exposes queryable region data (Phase 1), which enables multiple patterns: the provider can receive region descriptors directly, query them from a PR, or receive them via callback.
    - Tuning and kernel compilation may or may not be separable depending on the provider. TVM can compile without tuning (producing unoptimized kernels). Mirage's search may be integral to compilation. The kernel provider interface should not assume tuning is required, but also should not assume it is always separable.
- **RegionDescriptor design:** What information does a region descriptor contain? What is needed for a kernel provider to decide if it can handle a region, and to compile a kernel for it? Resolved when Phase 5 begins, informed by what Phase 1 learns about region representation.
