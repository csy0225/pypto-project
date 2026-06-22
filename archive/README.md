# Archive

Historical content frozen at archive cutoff. Once written, content
here is **append-only** — never rewrite history.

## Contents

| Doc | Span | Purpose |
|-----|------|---------|
| [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md) | Phase 01-19 (2026-05 to 2026-06-22) | Compressed history of pypto kernel prototype development |
| [`milestones-2026-Q2.md`](milestones-2026-Q2.md) | 2026-Q2 sessions | Session-by-session milestone log + pin snapshot history + resolved blockers |

## What goes where

| Trigger | Doc |
|---------|-----|
| Phase completion (the phase moves out of `phases/` into history) | Append a section to `prototype-phase-01-19-summary.md` or its successor |
| Session-end milestone summary | Append entry to `milestones-2026-Q2.md` |
| Blocker resolved | Add "Resolved blockers" entry to `milestones-2026-Q2.md` + remove from `../blockers.md` |
| Pin snapshot moves (any push to fork) | Append row to `milestones-2026-Q2.md` "Pin snapshot history" |

## Loose phase docs on dev host (NOT in this repo)

The 26 detailed phase doc files at
`<dev-host>/data/chensiyu/hw_project/pypto/docs/step3p5/phases/01-19*.md`
were intentionally **not migrated** into this repo. They contain
session-specific detail that's of historical reference value only; the
compressed summary in [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)
captures the essence.

If a future need arises (e.g., a contributor needs to trace why a
specific design decision was made), those files are still readable on
the dev host. They can be migrated piecewise into this archive on
demand without disrupting the live tracker.
