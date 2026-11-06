# Instructions

## Getting Started
First, clone the github repository for this repo:
```
git clone [TODO: add url here]
```

We strongly recommend using [uv](https://docs.astral.sh/uv/#uv) to easily reproduce our python environment and to run scripts. 
If you don't already have `uv` installed, install it using the instructions here: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/).
Once installed, you can easily reproduce the correct environment by running `uv sync` from within the repo directory.
To run a python script you can use the command `uv run <script_name.py>` or `uv run python <script_name.py>` from within the repo.
(In general you can replace the CLI command `python` with `uv run python` to automatically ensure the correct venv gets used.)

Then open `challenge.ipynb`. It contains the whole challenge.

## What is in the repo

| file | what it is |
|---|---|
| `challenge.ipynb` | the five problems; start here |
| `games.py` | payoff/classical-value helpers and the random games of Problem 5 |
| `check_validity.py` | validity checks and the repair step used in grading |
| `check_submission.py` | run this on your answers before you submit |
| `classical_values.npz` | our exact classical values for the Problem 5 games |

## Submitting

Produce one `.npz` file per problem, in a single directory:

| file | keys |
|---|---|
| `problem_1.npz` | `M`, `N` |
| `problem_2.npz` | `rho`, `M`, `N` |
| `problem_3.npz` | `V`, `rho`, `M`, `N` |
| `problem_4.npz` | `num_games_with_quantum_advantage` |
| `problem_5.npz` | `num_games`, `indices`, `rho`, `M`, `N` |

Check them first:
```
uv run python check_submission.py --directory .
```
This is the same code we grade with, so for Problems 1, 2, 3 and 5 the scores it prints are the scores you will get.
Problem 4 is only checked for format, since printing its score would give away the answer.

### How to hand in `problem_5.npz`

If `problem_5.npz` is under 100 MB, commit it like the other files.

If it is larger, git will reject it, so attach it to a GitHub Release in your repo instead.
A release is a named snapshot of your repo with files attached; the files are stored alongside the repo rather than in git, so the size limit is 2 GB per file.

From inside your repo run
```
gh release create submission problem_5.npz
```
(`gh` is the [GitHub CLI](https://cli.github.com/).)
Or in the browser: **Releases → Draft a new release**, enter the tag name `submission`, drag in `problem_5.npz`, and click **Publish release**.

To replace your submission with a newer one, run
```
gh release upload submission problem_5.npz --clobber
```
If you have a `submission` release, we grade the `problem_5.npz` attached to it as of the deadline; otherwise we grade the committed file.

## Rules

**Array conventions.** All matrices are `complex128` and row-major. A state `rho` has
Alice's subsystem first. POVMs are indexed `M[x, a]` and `N[y, b]`. Payoff tensors are
indexed `V[x, y, a, b]`. Never submit a pre-formed joint operator; submit Alice's and
Bob's measurement operators separately.

**Local dimension.** Alice's and Bob's systems may each have dimension at most 5, except
in Problem 1 where the state is given to you and the dimension is 2.

**Validity, then repair.** What you submit must already be a density matrix and POVMs to
within `1e-6`. Anything that fails that check scores nothing. 
Whatever passes is then projected onto the exactly valid
state and POVMs nearest to it, and only the repaired objects are scored. Repair leaves a
genuinely valid strategy untouched to machine precision; it exists only so that a rounding
error can never be cashed in as quantum advantage.

**Classical values are ours.** We compute every classical value exactly, by enumeration,
and those are the numbers you must beat. They are published for Problem 5 in
`classical_values.npz`. A Problem 5 witness has to clear its classical value by at least
`1e-9`. That margin exists only to stay clear of floating point: we evaluate a repaired
strategy to about `1e-15`, so anything clearing `1e-9` is a genuine advantage.

**The Problem 5 games are fixed.** They are generated from a published seed by SHA-256, so
they do not depend on your NumPy version or on the order you generate them in.
`games.fingerprint()` must print `cb33fab53cfd870b`; if it does not, something is wrong
with your checkout and you should tell us.

**Problem 5 limits.** You choose `N`, the number of games you take on, from 10, 100, 1000,
10,000 and 100,000, and you may submit at most 25,000 witnesses. `save_witnesses`
compresses the file; low-dimensional strategies compress well, but a large submission of
dimension-5 strategies can exceed 100 MB (see "How to hand in `problem_5.npz`" above).

**Scoring.**

| problem | score |
|---|---|
| 1 | 1 if your strategy wins CHSH more than 75% of the time, else 0 |
| 2 | the verified advantage over the classical value |
| 3 | the verified advantage of the game you submit |
| 4 | `max(0, 1 - abs(your answer - true count) / true count)` |
| 5 | `games.score(k, N)` with `k` your verified witnesses out of the `N` games you took on |

Problems carry equal weight. Ties on Problem 5 are broken by the total verified advantage
summed over your witnesses.

Note on Problem 5: the score is a valid lower confidence bound for any `N` fixed in
advance. Because you may choose `N` after seeing how you did, treat it as a leaderboard
score rather than a strict 95% interval. It stays conservative either way, since every
witness you fail to find only lowers it.

**Anything goes computationally.** Any algorithm, any library, any hardware. What we check
is the strategy you hand in, not how you found it.
