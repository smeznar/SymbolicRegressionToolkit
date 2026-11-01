---
name: review-module
description: Run the codebase quality review (docstrings, spec-first tests, mutation testing, bug fixes) on one SRToolkit module from the checklist in notes/notes.txt. Use when the user runs /review-module <path or checklist item>, or asks to review/continue the quality review of a module.
---

# review-module: quality review of one module

Argument: a module path (`SRToolkit/utils/serialization.py`) or a checklist item number from
`notes/notes.txt`. Without an argument, take the first unticked item of the checklist and confirm it
with the user.

## Before starting

1. Read the "Codebase quality review" section of `notes/notes.txt`: the checklist (including the
   findings noted next to this module), the per-module steps, and the gate commands.
2. Read the "Writing docstrings" and "Writing tests" sections of `CONTRIBUTING.md`.
3. Check the dependency order: if an earlier checklist item that this module imports is still
   unticked, say so, and let the user decide whether to continue.
4. Tell the user which module you are reviewing and which known findings apply to it.

Rules for the whole review:
- Explain changes and get the user's approval before editing; every decision about intended
  behaviour belongs to the user.
- Never commit.
- Run mutmut only with the capped command from the notes (`systemd-run ... MemoryMax=8G`,
  `--max-children 4`). When judging this module's test file alone, temporarily point
  `pytest_add_cli_args_test_selection` at it, and restore it to `["tests/"]` afterwards.
- One module per review. Some checklist items group several files; review those together.

## Steps (run the agents in module scope, and stop after each phase)

1. **Docs audit**: launch `doc-reviewer` (Phase 1) on the module. Relay its purpose lines and
   decisions to the user concisely, with your recommendation for each. Wait for their decisions.
2. **Docs rewrite**: continue `doc-reviewer` (Phase 2) with the decisions. Check that the code is
   unchanged (only docstrings and comments may change), and that doctests and lint pass.
3. **Test plan**: launch `spec-tester` (Phase 1). Relay the **test map** (reason | what it pins |
   test | expected value), the spec gaps and the doc-clarity items. Send spec gaps and wording fixes
   back to `doc-reviewer` before continuing.
4. **Black-box tests**: `spec-tester` Phase 2. Known bugs become strict xfails (one per bug). Relay
   any failure that isn't a known bug: it is a new bug, a spec bug or a test bug.
5. **Clear-box**: `spec-tester` Phase 3 (coverage, dead code, suspected bugs).
6. **Mutation**: `spec-tester` Phase 4. Relay the old-vs-new mutation scores and classified survivors.
7. **Prune and audit**: `spec-tester` Phase 5 (prune the new tests to the lean budget; keep, merge,
   rewrite or delete each old test). The user approves the lists.
8. **Apply**: `spec-tester` Phase 6. Coverage and mutation score must not drop.
9. **Fix the bugs** the xfails record (with the user's go-ahead). Remove each xfail marker as its
   bug is fixed, run the full suite, and rerun the capped mutmut on the module.

Verify each agent report yourself before relaying it: rerun the tests, check that the claimed
numbers hold, and spot-check derivations. If a report is long, summarise it, keeping every decision
the user has to make.

## Finish

1. Run the gates from the notes (ruff, the strict docstring check, the full suite with coverage, and
   `mkdocs build --strict` if docstrings changed).
2. Draft `CHANGELOG.md` lines (unreleased section, existing style) for anything a user would notice,
   and show them for approval.
3. Update `notes/notes.txt`: tick the module with a one-line result (tests, bugs fixed, mutation
   score). Move findings that belong to other modules next to those modules' checklist items.
4. Report: what changed, the bugs fixed, behaviour changes, and anything left open.
