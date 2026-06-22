# Architecture

Cross-repo design notes for the pypto step3p5 stack. These docs
describe how the 5 code repos fit together and how the Phase 2 vLLM
integration interfaces.

## Contents

| Doc | Purpose |
|-----|---------|
| [`overview.md`](overview.md) | Big-picture: 5 repos + vLLM, what each one does, where data flows |
| [`vllm-step3p5-mapping.md`](vllm-step3p5-mapping.md) | Operator-level mapping between vLLM's `Step3p5Model` and pypto's `decode_fwd` — required reference for Phase 20 monkey-patch implementation |

## When to add a new architecture doc

- **A new repo joins the project** → describe its role in `overview.md`
  + add a focused doc here if it has its own internal architecture.
- **A new cross-repo interface is introduced** → focused doc for that
  interface (e.g., the vLLM mapping doc).
- **A non-obvious data flow needs to be documented** → focused doc.

Avoid documenting per-repo internals here — those belong inside the
respective repo's `docs/`. This directory is for **cross-repo** design
content only.
