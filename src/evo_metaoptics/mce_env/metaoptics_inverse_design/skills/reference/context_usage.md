# Context Usage

How to use copied `context/` artifacts without treating them as ground truth.

## Reading context

- Treat `context/` as optional learned guidance from prior iterations.
- Reuse stable rules or examples when they fit the current query and `gt_eval`.
- Prefer concise synthesis over copying large context fragments verbatim.

## Safe usage pattern

1. Read `SKILL.md` first for the active contract and strategy.
2. Read only the reference files needed for the current failure mode or design goal.
3. Read `context/` artifacts last, then adapt them to the current sample.
4. Keep generated code deterministic and bounded even if context suggests broader search.
