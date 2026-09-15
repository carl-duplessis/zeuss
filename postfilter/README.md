# postfilter

Re-rank a candidate list by a logical exclusion constraint, applied **after**
scoring — not folded into whatever produced the scores in the first place.

```python
from postfilter import rerank, no_cycle_exclusions

# children: parent -> {entities one step below it in some directed hierarchy}
# here: animal -> mammal -> dog, so "dog" and "mammal" are both descendants of "animal"
children = {"animal": {"mammal"}, "mammal": {"dog"}}

# candidates: whatever your model already returned for "animal's parent category",
# as (name, score) pairs - the model's top pick ("dog") would create a cycle
candidates = [("dog", 0.42), ("organism", 0.40), ("entity", 0.38)]

excluded = no_cycle_exclusions(children, subject="animal")  # {"mammal", "dog"}
answer = rerank(candidates, penalty=lambda c: 10.0 if c in excluded else 0.0)[0]
# ("organism", 0.4) - "dog" was the model's top pick but can't also be its own ancestor
```

No dependency on any embedding model, graph library, or scoring function —
this only re-ranks `(name, score)` pairs you already have. Zero required
dependencies.

## Why this exists

Extracted from a real knowledge-graph completion project
([zeuss](https://github.com/carl-duplessis/zeuss)) that tested a specific,
narrow question with real data rather than assuming the answer: when a
logical constraint should veto a candidate, is it better to fold that
constraint *into* an iterative scoring/optimisation process, or to apply it
as a static re-rank of whatever that process already returned?

Across four independent axes — which relation was being predicted, how the
constraint was derived, which dataset it ran on, and what logical shape the
constraint took — the static re-rank (this package's approach) consistently
matched or beat folding the constraint into the scoring process itself, and
never did worse. Full experiments, numbers, and honest failure cases are in
that project's `docs/ROADMAP.md` (the "Post-closure" entries) and
`README.md`.

## What's validated, concretely

| Constraint shape | Source | Dataset | Real held-out result |
|---|---|---|---|
| No cycles in a directed hierarchy | Derived from the graph's own structure | WordNet (WN18RR) | BASELINE 10.6% → re-ranked 32.8%, n=198 |
| No cycles, reused across relations | Same, applied to relations it wasn't mined from | WordNet (WN18RR) | BASELINE 0–10% → re-ranked 32–80%, n=22/10 |
| Declared category disjointness | An external classification (WordNet's lexicographer files) | WordNet (WN18RR) | BASELINE 33.3% → re-ranked 44.0%, n=84 |
| No cycles in a containment hierarchy | Derived, different dataset entirely | Freebase (FB15k-237) | BASELINE 3.7% → re-ranked 7.3%, n=109 |
| Mined cross-relation implication | Support/confidence statistics over real triples | Freebase (FB15k-237) | BASELINE 34.8% → re-ranked 87.0%, n=69 |

Every one of these is a real, held-out measurement — not a synthetic or
hand-built case. See the source project for the harnesses these numbers came
from, including the honest failures along the way (two other datasets ruled
out because the constraint that seemed promising was either uninformative or
actively wrong).

## What's in this package

- `postfilter.rerank(candidates, penalty)` / `postfilter.top1(...)` — the
  generic primitive: `score * exp(-penalty(name))`, nothing more.
- `postfilter.no_cycle_exclusions(children, subject)` — builds the exclusion
  set for the "no cycles in a directed hierarchy" shape.
- `postfilter.disjoint_category_exclusions(category_of, disjoint_groups,
  subject, candidates)` — builds the exclusion set for the "declared category
  disjointness" shape.

The third validated shape (mining a cross-relation implication via
support/confidence statistics) isn't reimplemented here — it needs the
actual triples of your knowledge graph to mine from, which is domain-specific
enough that duplicating it outside of where those triples already live
wouldn't save real work.

## A real caveat, not smoothed over

A "hard" veto here is `penalty=10.0`, not literal exclusion — `exp(-10)`
times a small positive score can still beat `0.0` if every other live
candidate happens to score exactly zero. Measured directly: this happened in
5 of 198 real cases during validation, identically for both re-ranking and
the alternative it was compared against, so it didn't bias that comparison —
but it's a real floor on how "hard" an exponential-penalty veto can be. If
your application needs a true hard exclusion, filter vetoed candidates out
of the list before calling `rerank` instead of relying on the penalty alone.

## Install

```
pip install -e .
```

## Test

```
pip install -e ".[dev]"
pytest
```
