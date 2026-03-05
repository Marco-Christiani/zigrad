---
title: Architecture
---

# Architecture

Zigrad follows a staged model:

1. Program Representation (PR)
2. Interchange Module (IM, currently StableHLO/MLIR)
3. Backend compile + execute

For canonical terminology and boundaries, see `docs/DESIGN.md` in the repo.
