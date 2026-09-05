---
name: release-gate
description: Pre-release and pre-commit verification gate for rtldoc. Use before any commit to main, any version bump, tag, GitHub release, or PyPI publish, and whenever asked "is this safe to ship", "run the checks", or "can we release". Encodes which checks are blocking, which traps produce false confidence, and how to A/B a change against its parent commit.
---

# Release gate

A parser fails silently. Nothing crashes — text is just wrong, and nobody
notices until it is in a vector store. So the gate is not "does it run", it
is **"did I measure the thing I changed, at real scale, against a baseline"**.

## Blocking checks — all must pass

```bash
cd /Users/umerjavaid/Documents/umerwork/AI/pdf_parser

python3 eval/regression.py                  # cell-level table fixtures
python3 eval/invariants.py book/ corpus/    # property checks, whole corpus
```

**1. Golden fixtures: must be N/N, zero failures.**
These are cell-level table grids. They catch structural changes nothing else
will. If a fixture changes, either you broke it or the fixture was wrong —
decide which, in writing, before proceeding.

**2. The four HARD invariants must all be 0:**

| invariant | meaning if nonzero |
|---|---|
| crashes | a document class is unparseable |
| presentation forms out | encoding leak — the deshaping guarantee is broken |
| non-rectangular tables | a grid has ragged rows; downstream will misalign |
| non-deterministic pages | same input, different output — your index will drift |

**3. Soft metrics must not regress:**
- `mean letter coverage` — down means **text was lost**
- `mean letter excess` — up means **text was duplicated**
- `pages flagged low cov` / `high exc`

Soft metrics are the ones that catch real damage. Coverage dropping by 0.001
is a thousand lost letters at corpus scale.

## The trap that produces false confidence

**A soft metric moving is not proof you caused it.** The corpus composition
changes; other commits land. Always A/B against the parent commit on the
*same files*:

```bash
git checkout -q <parent-sha> -- rtldoc/
python3 -c "
import sys; sys.path.insert(0,'eval'); import invariants
r=invariants.check_pdf('book/<file>.pdf')
print('BEFORE', sum(r['coverage'])/len(r['coverage']), r['low_coverage'])"
git checkout -q HEAD -- rtldoc/
```

Real case: two "new" low-coverage pages appeared in a sweep. A/B showed
identical values (0.758, 0.804) on the parent — they surfaced only because
the corpus grew from 118 to 128 PDFs. Not regressions. Without the A/B that
would have blocked a good release, or worse, been "fixed".

`git stash` does **not** work for this if the change is already committed —
it silently stashes nothing. Use `git checkout <sha> -- <path>`.

## Sampling bias — the mistake that cost most

**Measure on the full corpus, not a sample.** A 40-document sample reported
heading quality at 7% unusable. The full 1,462-document audit said **17.1%**
— and the sample had hidden the *dominant* failure mode entirely (4
occurrences in the sample vs **1,442** corpus-wide).

Samples are for *finding* bugs fast. They are not for *deciding* a bug is
fixed. Before claiming a fix works, run the full corpus.

## Scope-aware checks

Match the check to what the change touches:

| changed | must additionally verify |
|---|---|
| `primitives.py` / `geobidi.py` | **everything** — these feed all documents. Check a large English doc AND the Arabic guide |
| Arabic / bidi logic | `book/BilArabi_TG07.pdf`: coverage, 0 leaks, and the specific corruption counts you targeted |
| table detection | golden fixtures are primary; also table counts on a known financial filing |
| heading / role typing | full-corpus heading audit, not a sample |
| anything with a threshold | show the two populations the threshold separates, with numbers |

Arabic is the canary for glyph-level changes: it exercises bidi, ligatures,
shaping and RTL ordering at once. A change that is safe there is usually safe.

## Release steps

Only after every check above is green:

```bash
# 1. bump
sed -i '' 's/^version = "X.Y.Z"/version = "X.Y.Z+1"/' pyproject.toml
# update the git-install pin in README.md to the new tag too

# 2. commit, tag, push
git commit -am "Bump version to X.Y.Z+1"
git tag -a vX.Y.Z+1 -m "..."
git push origin main && git push origin vX.Y.Z+1

# 3. release — THIS triggers the PyPI publish workflow
gh release create vX.Y.Z+1 --title "..." --notes "..."

# 4. confirm
gh run list --workflow=publish.yml --limit 1
curl -s https://pypi.org/pypi/pdf-rtldoc/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
```

**A published version number can never be reused or truly withdrawn.**
Confirm the version with the user before creating the release.

Patch vs minor: bug fixes only → patch. Anything that changes **output
structure** for a class of documents (new detector, new block role, changed
reading order) → minor, even if it is "just a fix", because consumers will
see different output.

## Release notes

State the root cause, not the symptom. "Fixed Arabic text" is useless;
"the ligature's alef is emitted zero-width and the following lam at double
width, so ordering by position alone inverted every lam-alef pair" tells a
user whether their documents are affected. Include the before/after numbers
and the verification scope. Keep a known-limits section — shipping with
honest limits beats shipping with silent ones.
