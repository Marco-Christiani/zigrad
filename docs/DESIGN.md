---
created_at: '2026-01-29T22:38:15.704899'
id: a63022e6-ea36-4bf7-a3ab-1131d602961f
phase: mainline
scope: design
status: active
title: DESIGN
updated_at: '2026-02-16T00:00:00.000000+00:00'
---

**Purpose:** Define the stable architecture that supports multiple toolchains (XLA/PJRT, IREE, TVM/microTVM, vendor dispatch, Mirage, MPK). Intended to be stable by--in part--avoiding implementation details, to keeping churn controllable.

---
## 0) Vocabulary

### 0.1 Core Concepts

- **Program Representation (PR)**: The in-memory data structure representing a program's semantics.
    - Used as the substrate for high-level transformations and analysis.
    - Has a debug pretty-print view (ZXPR). Not a serialization format, not an interchange format.
    - Internal to Zigrad. PR is a data structure, not an IR.

- **Interchange Module (IM)**: A serialized artifact intended to be consumed by an external compiler toolchain.
    - Current default: a **StableHLO/MLIR profile** (as defined by the chosen PS/backend).
    - The boundary between zigrad and the backend. Zigrad does not run passes on an IM.

- **Executable Artifact (EA)**: Output of a backend that can be executed.
    - Examples: PJRT executable, IREE VM module, TVM runtime module, custom bare-metal image.

- **Kernel Artifact (KA)**: A compiled kernel for a specific target, inducing an implementation boundary in compilation/execution.
    - KA invocation uses a well-defined internal **Kernel Calling Contract** (KABI).

- **Backend**: The atomic unit responsible for compilation and execution of a computation on a specific target.
    - Owns device enumeration, buffer management, compilation (IM -> EA), and execution (EA -> results).
    - Compilation and execution are grouped under one interface because in practice they are shipped together (e.g., PJRT is a single shared library). They remain distinct stages with different assumptions and dependencies.
    - Backends may support compilation, execution, or both. Dependency requirements should follow capability boundaries: if only execution is needed, compiler-only dependencies should not be required.
    - The **compilation** path consumes an IM and produces an EA. It may include optimization, scheduling, code generation, and linking/packaging steps. It may expose extension points (e.g., custom passes/dialects, custom call lowering).
    - The **execution** path loads and runs an EA on a target system. It defines execution boundary contracts (buffer mutability/ownership, synchronization model, device enumeration, resource limits).

- **Kernel Provider**: An extension component that can claim PR regions and produce KAs for specific targets.
    - Kernel providers do not own the full compilation/execution pipeline. They produce KAs invoked by the backend at execution time via KABI.

- **Pipeline Specification (PS)**: A concrete end-to-end configuration selecting a backend, kernel providers, and passes.
    - Defines how a program is transformed, realized into IM, compiled into EA, and executed.
    - Any deeper customization is expressed via extension points (e.g., passes/dialects/custom calls) and/or annotations.

### 0.2 Annotations and Information Passing

**Annotation**: Metadata attached to regions, operations, or values intended to influence transformation or compilation behavior while preserving program semantics.

Annotations are classified by intent:

- **Semantic annotations**: Constrain or define program meaning and legality.
    - Examples: aliasing/view relationships, effect ordering, donation legality, materialization requirements.
    - Must be respected across all transformations and realizations.

- **Steering annotations**: Influence how a backend realizes or optimizes a program, without affecting correctness.
    - Examples: "eligible for kernelization", "prefer custom lowering", "avoid fusion across this boundary".
    - Steering annotations **must be safe to ignore** unless they are realized as explicit constructs (e.g., custom calls, preserved boundaries).

- **Debug annotations**: Carry provenance or diagnostic information only.
    - Examples: naming, profiling tags, source locations.

Annotations may be realized and transported through a pipeline using one or more **carriers**, depending on the PS/backend:
- PR-level metadata (internal to Zigrad)
- IM-level attributes (backend-visible)
- Explicit constructs (e.g., custom dialect ops, custom calls)
- Backend configuration (out-of-band)

*Aside: Custom dialects and FFI/custom calls are stronger extension mechanisms that either (a) require toolchain passes to lower the dialect into accepted constructs, or (b) create explicit call boundaries that the backend must preserve and implement.*

---

## 1) End-to-End Model

### 1.1 Stage Table

| Stage | Name                      | Ownership | Responsibilities                                                                                                                 |
| ----: | ------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------- |
|     1 | User Program (Zig / MLIR) | User      | Define computation, optionally provide manual constraints / features.                                                            |
|     2 | PR                        | Zigrad    | Meaning-preserving transformations (AD, legality), normalization, region identification, annotations (internal).                  |
|     3 | IM                        | Shared    | Backend-facing serialized artifact representing the computation to compile.                                                      |
|     4 | Backend                   | Mixed     | Compile IM into EA, execute EA: device mgmt, buffers, synchronization, profiling.                                                |

### 1.2 Optionality Boundaries

Where controlled variation and specialization are allowed, without destabilizing the core model.

**1) Transformation Boundary (PR -> PR)**
Program-model-level transformations close to the domain that preserve or intentionally extend meaning:
- Autodiff (VJP / JVP)
- Legality reasoning (aliasing, views, buffer donation)
- Algebraic normalization and canonical rewrites
- Region identification and annotation (for specialization / kernel providers)

**2) Realization (Lowering) Boundary (PR -> IM)**
Realization of a PR into a backend-facing IM.
- IM form is defined by the selected PS/backend.
- StableHLO/MLIR is the current default IM for MLIR-based backends.
- Backends may accept or introduce additional dialects (existing or custom), provided they are fully resolved or lowered before producing an EA.

**3) Kernelization Boundary (PR subgraph -> KA, referenced from IM)**
Optional specialization step where selected regions are replaced by KA boundaries.
- Kernelization is **best-effort** and must not be required for correctness.
- If a KA cannot be produced, validated, or invoked, the pipeline must fall back to the baseline realization path defined by the PS/backend.
- KA boundaries constitute explicit implementation boundaries that the backend must preserve (or explicitly fail).

*Note: Possible mechanisms include multi-level search/superoptimization, schedule search, explicit vendor/library dispatch.*

---
## 2) PR

### 2.1 PR Responsibilities
Normative intent.

PR must be sufficient to:
- represent program meaning with a small set of typed primitives
- support program transformations (at least AD)
- support legality reasoning:
    - alias/view relations (what shares storage)
    - buffer donation legality (when in-place reuse is safe)
    - minimal effect ordering
- support **region identification** for optional specialization
- realize cleanly to IM with explicit boundary contracts

### 2.2 PR Non-Responsibilities
Explicit non-goals to avoid violating the design.

PR does NOT:
- choose schedules (tiling, pipelining, vectorization)
- pick kernels except via explicit annotations/extension points or via kernelization boundaries
- define device placement or multi-device partitioning (future extension)
- define runtime memory allocation strategies (backend responsibility)

---

## 3) Lowering Contract: PR -> IM
Note: StableHLO is default.

### 3.1 Contract

Before lowering PR to an IM profile:
- all operations in the lowered region must have defined semantics and a lowering rule
- any "custom op" must be resolved to:
    - a decomposition into IM ops, or
    - a KA boundary (e.g., a custom-call style boundary in the IM)
- legality-sensitive annotations (donation eligibility, materialization barriers) must be finalized or explicitly dropped (with diagnostics)

### 3.2 What Crosses the Boundary

PR -> IM carries:
- computation structure (ops + dependencies)
- shapes/dtypes (policy: prefers shape-instantiated programs, polymorphism handled outside PR)
- **debug annotations** (for dumps, provenance, profiling)
- **steering hints/annotations**

Notes:
- No steering annotation is assumed to be honored unless validated for the selected PS/backend.
- If steering intent must be enforced, it must be realized as an explicit construct (e.g., a preserved boundary such as a custom call).

---

## 4) Backend and Pipeline

### 4.1 Backend

A backend is the atomic unit responsible for compilation and execution of a computation on a specific target. Conceptually, a backend provides:

- `compile(IM, options) -> ExecutableArtifact`
- `execute(EA, inputs, exec_options) -> results`
- `get_devices() -> device list`
- `transfer(device, buffer, direction) -> transferred buffer`

**Notes:**
- The backend owns device interaction, synchronization, buffer allocation/deallocation, and (where applicable) loading of compiled artifacts.
- A backend may support compilation, execution, or both. The dependency requirements for each capability should be independent: a deployment that only executes pre-compiled artifacts should not require compiler-only dependencies.
- The backend surface may vary by environment class (hosted vs embedded Linux vs freestanding).

### 4.2 Pipeline Relationship

The pipeline is the pass chain that transforms PR into IM. The backend is a peer, not a subordinate of the pipeline. The pipeline ends at IM. Compilation and execution are backend responsibilities. The coordinator (user code or a thin driver) calls both. This separation keeps the pipeline as a pure program transformation system and the backend as a pure hardware interaction system.

This model supports both JIT (pipeline + compile + execute in one process) and AOT (pipeline + compile on build machine, execute on target) without requiring the pipeline to understand backend lifecycle.

**Open question - kernel provider tuning placement:** Search-based kernel providers (TVM MetaSchedule, Mirage) perform expensive hardware-dependent tuning. Where tuning occurs (within the pipeline, between pipeline and backend, or otherwise) is subject to further consideration.

### 4.3 Pluggable Backend Policy

Backend implementations are pluggable components. Systems designed by their upstream projects to be dynamically loadable (e.g., PJRT) must be integrated as runtime-loadable implementations rather than compile-time dependencies. Kernel providers follow the same constraint.

The architectural goals are:
1. The zigrad core builds without requiring backend SDKs.
2. Backend integrations can be enabled or disabled without rebuilding zigrad.
3. Backend-specific headers and types do not appear in core layers (PR, passes, pipeline).
4. The core system depends only on the declared backend and kernel provider interfaces.

### 4.4 Boundary Contracts (Execution Boundary)

We require a defined internal contract for the execution boundary:
- **Buffer mutability**: which inputs/outputs are read-only vs writable
- **Donation/consumption**: whether the backend honors donation opportunities (pipeline-specific; safe to ignore)
- **Alignment/layout expectations**: when invoking Kernel Artifacts (KA), the pipeline defines the kernel calling contract (KABI)
- **Synchronization semantics**: minimal guarantee that `execute` respects explicit dependencies supplied via `exec_options` (pipeline-specific extension)

---

## 5) Kernel ABI
KABI defines an internal calling contract for kernel invocation; it does not imply external stability guarantees.

### 5.1 Purpose

A well-defined Kernel Calling Contract (KABI) enables KA boundaries such as:
- vendor library dispatch (Accelerate, oneDNN, cuBLAS)
- custom kernels
- Mirage/TVM-generated kernels
- bespoke accelerator kernels

### 5.2 Requirements

Kernel calls must define:
- dtype(s), shape(s), rank(s)
- layout/stride policy
- input/output buffers and mutability/donation rules
- target device kind + stream/sync contract


