---
created_at: '2026-01-29T22:38:15.704899'
id: a63022e6-ea36-4bf7-a3ab-1131d602961f
phase: mainline
scope: design
status: active
title: DESIGN
updated_at: '2026-02-10T00:28:26.808184+00:00'
---

**Purpose:** Define the stable architecture that supports multiple toolchains (XLA/PJRT, IREE, TVM/microTVM, vendor dispatch, Mirage, MPK). Intended to be stable by--in part--avoiding implementation details, to keeping churn controllable.

---
## 0) Vocabulary

### 0.1 Model vs Representation vs Artifact vs Contract

- **Program Model (PM)**: Zigrad-owned conceptual model of a program's semantics.
    - Defines the ontology, invariants, and relationships required for reasoning and transformation (e.g., autodiff, legality, normalization).
    - Exists independently of any concrete encoding or file format.

- **Program Representation (PR)**: The in-memory representation of a Program Model instance.
    - Used as the substrate for coarse high-level transformations and analysis.
    - Has a **pretty-print view** for debugging and inspection.
    - Internal to Zigrad, not an interchange format.

- **Interchange Module (IM)**: A serialized artifact intended to be consumed by external compiler toolchains.
    - Planned default IM: a **StableHLO/MLIR profile** (as defined by the chosen PS/toolchain).

- **Executable Artifact (EA)**: Output of a backend toolchain that a runtime can execute.
    - Examples: PJRT executable, IREE VM module, TVM runtime module, custom bare-metal image.

- **Kernel Artifact (KA)**: A specialized kernel inducing an implementation boundary in compilation/execution.
    - KA invocation uses a well-defined internal **Kernel Calling Contract** (KABI).

- **Toolchain**: The compilation path that produces an EA for a given target environment.
    - Conceptually: consumes an IM (and may include a PR->IM realization step).
    - May include optimization, scheduling, code generation, and linking/packaging steps.
    - May expose extension points (e.g., custom passes/dialects, custom call lowering).

- **Runtime**: The execution environment that loads (if applicable) and executes an EA on a target system.
    - Defines execution boundary contracts (buffer mutability/ownership, synchronization model, device enumeration, resource limits).
    - Provides the operational interface for running compiled programs.

- **Pipeline Specification (PS)**: A concrete end-to-end selection of a configured Toolchain + Runtime for a target environment.
    - A PS defines how a program is transformed, realized into IM, compiled into EA, and executed by the runtime.
    - Any deeper customization is expressed via extension points (e.g., passes/dialects/custom calls) and/or annotations.

### 0.2 Annotations and information passing

**Annotation**: Metadata attached to regions, operations, or values intended to influence transformation or compilation behavior while preserving program semantics.

Annotations are classified by intent:

- **Semantic annotations**: Constrain or define program meaning and legality.
    - Examples: aliasing/view relationships, effect ordering, donation legality, materialization requirements.
    - Must be respected across all transformations and realizations.

- **Steering annotations**: Influence how a toolchain realizes or optimizes a program, without affecting correctness.
    - Examples: "eligible for kernelization", "prefer custom lowering", "avoid fusion across this boundary".
    - Steering annotations **must be safe to ignore** unless they are realized as explicit IR constructs (e.g., custom calls, preserved boundaries).

- **Debug annotations**: Carry provenance or diagnostic information only.
    - Examples: naming, profiling tags, source locations.

Annotations may be realized and transported through a pipeline using one or more **carriers**, depending on the PS/toolchain:
- PR-level metadata (internal to Zigrad)
- IM-level attributes (toolchain-visible)
- Explicit IR constructs (e.g., custom dialect ops, custom calls)
- Toolchain configuration (out-of-band)

- *Aside: In general,  custom dialects and FFI/custom calls are stronger extension mechanisms that either (a) require toolchain passes to lower the dialect into accepted IR, or (b) create explicit call boundaries that the toolchain must preserve and the runtime must implement.*

---

## 1) End-to-End Model

### 1.1 Stage Table

| Stage | Name                      | Ownership | Responsibilities                                                                                                                  |
| ----: | ------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------- |
|     1 | User Program (Zig / MLIR) | User      | Define computation, optionally provide manual constraints / features.                                                             |
|     2 | PR                        | Zigrad    | Meaning-preserving transformations (AD, legality), normalization, region identification, annotations (internal).  Conforms to PM. |
|     3 | IM                        | Shared    | Toolchain-facing serialized artifact representing the computation to compile.  (algorithm-level, aka what to compute)             |
|     4 | Toolchain                 | Mixed     | Compile IM into EA (fusion/scheduling/codegen)                                                                                    |
|     5 | Runtime                   | Mixed     | Execute EA: buffers, synchronization, device mgmt, profiling                                                                      |

### 1.2 Optionality Boundaries

Where controlled variation and specialization are allowed, without destabilizing the core model.

**1) Transformation Boundary (PR -> PR)**
Program-model-level transformations close to the domain that preserve or intentionally extend meaning:
- Autodiff (VJP / JVP)
- Legality reasoning (aliasing, views, buffer donation)
- Algebraic normalization and canonical rewrites
- Region identification and annotation (for toolchain extension points / specialization)

**2) Realization (Lowering) Boundary (PR -> IM)**
Realization of a PR into a toolchain-facing IM.
- IM form is defined by the selected PS/toolchain.
- StableHLO/MLIR is the current default IM for MLIR-based toolchains.
- Toolchains may accept or introduce additional dialects (existing or custom), provided they are fully resolved or lowered before producing an EA.

**3) Kernelization Boundary (PR subgraph -> KA, referenced from IM)**
Optional specialization step where selected regions are replaced by KA boundaries.
- Kernelization is **best-effort** and must not be required for correctness.
- If a KA cannot be produced, validated, or invoked, the pipeline must fall back to the baseline realization path defined by the PS/toolchain.
- KA boundaries constitute explicit implementation boundaries that the toolchain must preserve (or explicitly fail).

- *Note: Possible examples of mechanisms: multi-level search/superoptimization, schedule search, explicit vendor/library dispatch.*

---
## 2) PR

### 2.1 PR Responsibilities
Normative intent.

PR must be sufficient to:
- represent program meaning with a small set of typed primitives
- support program transformations (at least AD in v0)
- support legality reasoning:
    - alias/view relations (what shares storage)
    - buffer donation legality (when in-place reuse is safe)
    - minimal effect ordering (NOTE: v0: keep minimal, expand later if needed)
- support **region identification** for optional specialization
- realize cleanly to IM with explicit boundary contracts

### 2.2 PR Non-Responsibilities
Explicit non-goals to avoid violating the design.

PR does NOT:
- choose schedules (tiling, pipelining, vectorization)
- pick kernels except via explicit annotations/extension points or via kernelization boundaries
- define device placement or multi-device partitioning in v0 (future extension)
- define runtime memory allocation strategies (runtime/toolchain responsibility)

---

## 3) Lowering Contract: PR -> IM
Note: StableHLO is default
### 3.1 Contract

Before lowering PR to an IM profile:
- all operations in the lowered region must have defined semantics and a lowering rule
- any "custom op" must be resolved to:
    - a decomposition into IM ops, or
    - a KA boundary (e.g., a custom-call style boundary in the IM)
- legality-sensitive annotations (donation eligibility, materialization barriers) must be finalized or explicitly dropped (with diagnostics)

### 3.2 What crosses the boundary

PR -> IM carries:
- computation structure (ops + dependencies)
- shapes/dtypes (NOTE: policy: v0 prefers shape-instantiated programs, polymorphism handled outside PR)
- **debug annotations** (for dumps, provenance, profiling)
- **steering hints/annotations**

Notes:
- No steering annotation is assumed to be honored unless validated for the selected PS/toolchain.
- If steering intent must be enforced, it must be realized as an explicit IR construct (e.g., a preserved boundary such as a custom call).


---

## 4) Toolchain and Runtime Interfaces

### 4.1 Toolchain Interface (compilation)
**Purpose:** Compiles artifacts.

Conceptually, a toolchain provides:
- `compile(IM or PR, options) -> ExecutableArtifact`
- `describe_inputs() -> {accepted IM profiles, supported dtypes/features, required metadata}` (optional)

**Notes:**
- The toolchain may perform fusion/scheduling/codegen internally depending on the pipeline.
- The toolchain does not allocate runtime buffers or manage device execution state beyond what is needed to produce the executable artifact.

### 4.2 Runtime Interface (execution)
**Purpose:**  Provide an operational interface for running EAs.

Conceptually, a runtime provides:
- `load(ExecutableArtifact) -> LoadedExecutable` (optional if EA is directly runnable)
- `execute(LoadedExecutable, inputs, exec_options) -> outputs`
- `query_capabilities() -> {devices, memory limits, supported dtypes, concurrency/sync model, etc.}`

**Notes:**
- The runtime owns device interaction, synchronization, and (where applicable) buffer allocation/deallocation policies.
- The runtime surface may vary by environment class (hosted vs embedded Linux vs freestanding).

### 4.3 Boundary Contracts (execution boundary)

We require a defined internal contract for the runtime execution boundary:
- **Buffer mutability**: which inputs/outputs are read-only vs writable
- **Donation/consumption**: whether the runtime/toolchain honors donation opportunities (pipeline-specific; safe to ignore)
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

---

## 6) Specialization Profiles (wip)

A "profile" is a user-selected compilation target that chooses:
- which PS(s) to use (XLA vs IREE vs TVM vs vendor dispatch)
- whether to enable kernelization (Mirage/Ansor) and under what constraints
- preferred cost model goals (latency vs throughput vs memory)

Profiles are described separately as thinner, change-tolerant documents.

