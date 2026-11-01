---
name: dev-check
description: Run the spec-first development workflow on the current changes to SRToolkit (new code, bug fixes, changed behaviour). Use when the user runs /dev-check, or asks to check, test or document what they changed, before they commit.
---

# dev-check: spec-first workflow for the current changes

Apply the project's testing and documentation rules (`CONTRIBUTING.md`: "Writing docstrings" and
"Writing tests") to what changed, not to whole modules. The `doc-reviewer` and `spec-tester` agents
do the work, each run with a **list of functions** (their scoped mode). Rules that always apply:

- Explain what you will do and get the user's approval before editing files.
- Never commit; the user reviews and commits.
- Run mutmut only through the capped command (`systemd-run --user --scope -p MemoryMax=8G
  -p MemorySwapMax=0 mutmut run --max-children 4 ...`).

## 1. Find what changed

1. Collect the changes: `git diff master...HEAD` plus uncommitted work (`git diff HEAD`, and
   `git status` for new files). If the user names a base or specific files, use those.
2. List the **changed public names** under `SRToolkit/`, grouped by module: functions, classes and
   methods whose signature, docstring or body changed, and new ones. Skip private helpers unless they
   are the whole change; they are tested through the public names that use them.
3. Show the list. If nothing under `SRToolkit/` changed, say so and stop.

## 2. Classify the change

Propose a classification for each module's changes with a one-line reason, and let the user confirm
or correct it (AskUserQuestion):

- **Refactor**: no behaviour change intended.
- **Bug fix**: the code violated its spec, or behaved wrongly where the spec was silent.
- **Feature / behaviour change**: new public names, or intended changes to what existing ones do.

## 3. Run the matching steps

**Refactor**
1. Run the module's tests and then the full suite.
2. If any existing test had to change, say so explicitly: either the behaviour changed (so this is
   not a refactor; reclassify) or that test checked the implementation rather than behaviour (propose
   rewriting it).

**Bug fix**
1. A failing test comes first, with the reason **regression**. Run `spec-tester` on the affected
   names, giving it the bug description, and limit it to Phase 2 for that one bug. Confirm that the
   test fails on the unfixed code. If the fix is already in the working tree, check this with
   `git stash` (and `git stash pop` afterwards), or explain why it can't be checked.
2. Fix, or confirm the existing fix, then run the test and the module's tests.
3. If the docstring was silent about the case, run `doc-reviewer` on those names so the spec states
   the intended behaviour.

**Feature / behaviour change**
1. **Docstrings first.** Run `doc-reviewer` (Phase 1) on the changed names. For code not written yet,
   it checks that each docstring is complete enough to test from. Bring its decisions to the user,
   then continue it for the rewrite.
2. **Test plan.** Run `spec-tester` Phase 1 on the changed names. Show the user the **test map**
   (reason | what it pins | test | expected value). This is the review step that matters most; wait
   for approval.
3. `spec-tester` Phase 2 writes the tests. Before implementation they are expected to fail ("red").
4. When the implementation is there: `spec-tester` Phases 3–4 (coverage and mutation on the changed
   names only). Bring surviving mutants and spec gaps to the user.

## 4. Gates (all kinds of change)

For each touched module:
```
ruff check SRToolkit tests && ruff format --check SRToolkit tests
ruff check <module> --isolated --preview --select D,DOC --config 'lint.pydocstyle.convention="google"'
pytest -q -p no:cacheprovider
mkdocs build --strict        # only if docstrings changed
```
Report only the strict docstring findings for the changed names; known false positives (exceptions
raised by a callee or a helper) are fine. Then:
- Suggest `/code-review` on the diff for a bug hunt the tests may not cover.
- If a user would notice the change (new feature, fix, breaking change), draft a `CHANGELOG.md` line
  under the unreleased section, matching the existing style, and show it for approval.

## 5. Report

Finish with a short summary: what changed, the tests added (count, plus the test map if new), bugs
found, gate results, and anything left open (spec gaps, surviving mutants, a CHANGELOG draft).
