# Version control

## What is tracked, and why

| | tracked | reason |
|---|---|---|
| source, tests, docs, configs | yes | the artifact |
| `runs/**/history.json` | **yes** (173 files, ~6 MB) | every evaluation of every run. "Which code produced table 3?" has to be answerable, and re-aggregating old runs needs no retraining |
| `runs/**` weights (`*.pt`) | no | ~1.3 GB, regenerable |
| `runs/_*/`, `runs/*smoke*/` | no | pytest fixtures and CPU smoke runs; no claim rests on them |
| `data/demos/` | no | regenerable with `scripts/make_demos.py` |

### The `.gitignore` shape is deliberate

The obvious spelling does not work:

```gitignore
runs/                      # excludes the directory
!runs/**/history.json      # ...and this silently does nothing
```

Git cannot re-include a file whose **parent directory** is excluded, so that
pair tracks zero history files while reading as though it tracks all of them.
The working version excludes the contents and keeps the directories:

```gitignore
runs/**
!runs/**/
!runs/**/history.json
```

## Branching

`main` is always green: it should pass `pytest -q -m "not slow"` at every
commit. Work happens on branches.

```
main                      always green, always reproducible
  feat/<thing>            new capability        feat/qmix-calibration
  fix/<thing>             defect                fix/eval-window-iter0
  exp/<thing>             experiment, may be abandoned   exp/maze-expert
  docs/<thing>            documentation only
```

Merge to `main` with a pull request even working alone. The PR body is where
the result goes, and CI runs the slow trainer smoke tests on PRs but not on
every push.

## Commit messages

Conventional Commits, because the prefix makes `git log --oneline` scannable
when looking for the change behind a number.

```
<type>(<scope>): <imperative summary under 72 chars>

Why, not what -- the diff already says what. For anything touching a
measurement, state the number before and after.
```

Types: `feat` `fix` `perf` `refactor` `test` `docs` `chore` `exp`.

A real one from this project:

```
fix(eval): report a window of evaluations, not the final one

Consecutive evals of one converged run differ by 0.20-0.30 on success.
Both ablation scripts took history[-1], i.e. one draw from that
distribution per seed. Phase 5's headline moves from -25% to -4% on the
same data, with no retraining.
```

## Tagging results

Every number that reaches a paper gets a tag, so the claim resolves to a
commit:

```bash
git tag -a result/phase7-shuffled -m "full vs shuffled, 16 agents, 5 seeds, p=0.694"
git push origin --tags
```

Naming: `result/<what>` for a measurement, `paper/<venue>-<date>` for a
submitted manuscript state.

## Rules that exist because they were broken here

1. **Never commit a command you have not run.** A `--eval-every` flag that
   did not exist shipped in written instructions once.
2. **A measurement change is a `fix`, not a `chore`,** and its message carries
   the before and after. Three headline numbers in this project moved after
   the measurement was corrected rather than the method.
3. **Tag before re-running.** Re-running an ablation with a changed default
   silently redefines what an old number meant.

## Working from Windows and from a Linux shell

Both touch the same tree, so `.gitattributes` pins `eol=lf` in the repository
and `core.autocrlf` is `false`. Without it the two disagree about every line
of every file and the first cross-shell commit is a whole-repo diff.

## Authorship

Commits and pull requests carry the people who did the work. No tool,
assistant or generator is named as an author, co-author or in a trailer.
`git commit` takes `-F <file>` here precisely so the message is exact bytes
chosen in advance rather than whatever an editor or tool appends.

Check before the first push of a fresh repository:

```powershell
git log --format='%an|%ae|%b' | Select-String -Pattern 'Co-Authored|Generated with|noreply@'
```

Silence is the pass. A match means rewrite the history now, while it is one
commit and has not left the machine.

## Publishing to GitHub

### Decide visibility first

The paper is under review. If the venue is double-blind, a public repository
carrying the author names de-anonymises the submission, and a reviewer who
finds it may be obliged to flag it. **Start private.** Making a repository
public later is one click; un-publishing a repository that has already been
cloned, forked or indexed is not possible.

Private also costs nothing here: GitHub Actions runs on private repositories,
and a private repo can be shared with named collaborators.

### Steps

```powershell
cd E:\MARL2.0\mapf_v2

# 1. Fresh repository. Pin line endings before the first add, or the first
#    commit from the other shell rewrites every file.
git init -b main
git config core.autocrlf false
git config user.name  "Amit Reddy"
git config user.email "ar0098@srmist.edu.in"

# 2. Stage everything the ignore rules allow, then CHECK before committing.
git add -A
(git diff --cached --name-only).Count        # expect 250
git diff --cached --name-only | Select-String '\.pt$'   # must print nothing
git diff --cached --name-only | Select-String 'demos/|\.venv/'  # nothing

# 3. Commit from the prepared file, so the message is exactly these bytes.
git commit -F E:\MARL2.0\initial-commit-msg.txt

# 4. Verify authorship before anything leaves the machine.
git log --format='%an|%ae|%b' | Select-String -Pattern 'Co-Authored|Generated with|noreply@'
#    No output = clean. Any output = stop and rewrite.

# 5. Create the repository, PRIVATE, with no README/licence/gitignore on
#    GitHub's side -- those would create a conflicting first commit.
gh repo create budget-mapf --private --source=. --remote=origin

# 6. Push.
git push -u origin main

# 7. Tag the states that numbers resolve to.
git tag -a paper/ieee-2026-10-18   -m "IEEE submission, 18 Oct 2026"
git tag -a result/shuffled-control -m "full vs shuffled: p=0.694 / 0.817 / 0.689 at 16 / 32 / 64 agents"
git tag -a result/qmix-negative    -m "QMIX cannot carry the ablation; FINDINGS-phase6b.md section 20"
git push origin --tags
```

Then delete `E:\MARL2.0\initial-commit-msg.txt`; it sits outside the repo so
it is never staged, and it has no further use.

### Settings worth setting once

| setting | value | why |
|---|---|---|
| Branch protection on `main` | require PR, require `tests` to pass | `main` is the branch the paper's numbers resolve to |
| Allow auto-merge | off | nothing here is urgent enough to merge unwatched |
| Actions permissions | read-only token | CI only reads and runs tests |
| Delete head branches on merge | on | `exp/` branches accumulate |

Branch protection on a private repository needs GitHub Pro or an
organisation. Without it, the same discipline is manual: still open the PR,
still wait for CI.

### Two files to settle before pushing

- **`LICENSE`** -- MIT, copyright Amit Reddy. SRM may claim rights in work
  produced under its affiliation; check before publishing under a permissive
  licence. No licence at all means all rights reserved, which blocks the
  reuse a research repository normally wants.
- **`CITATION.cff`** -- author order follows the paper. The
  `repository-code:` line still has a `<your-username>` placeholder.
