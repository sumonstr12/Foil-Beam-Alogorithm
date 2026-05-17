# FOIL and Beam Search Algorithm Implementation on Family Relations Dataset

## 1. Cover Page

**Assignment Title:** FOIL and Beam Search Algorithm Implementation on Family Relations Dataset
**Student Name:** Sumon
**ID:** 220212
**Course:** Machine Learning
**Topic:** Learning First Order Logic / Rule Learning
**Department/Discipline:** Computer Science and Engineering
**University:** Khulna University
**Submitted To:** Course Teacher
**Submission Date:** May 2026

---

## 2. Abstract

This assignment implements simplified FOIL and Beam Search algorithms on a family relations dataset. The background predicates are `father(X,Y)` and `mother(X,Y)`. The target predicate is `grandfather(X,Y)`. Python is used to prepare positive and negative examples, calculate FOIL Gain, run Beam Search, and generate output files. The final learned rules are:

```prolog
grandfather(X,Y) :- father(X,Z), father(Z,Y).
grandfather(X,Y) :- father(X,Z), mother(Z,Y).
```

---

## 3. Introduction

Rule learning is a machine learning method where the output is a human-readable logical rule. In first-order logic, rules can use variables and relations between objects. For example, `grandfather(X,Y)` means X is the grandfather of Y.

The family relations problem is suitable for this assignment because it can be represented naturally using logical facts. Parent-child relationships are represented by `father(X,Y)` and `mother(X,Y)`. From these basic facts, complex relations like `grandfather` can be derived using learned rules.

This type of learning is called **Inductive Logic Programming (ILP)** — the algorithm is given background knowledge (father, mother facts) and examples (positive and negative), and it learns Prolog-style rules automatically.

---

## 4. Objective

The main objectives are:

- Implement the FOIL algorithm for the family dataset.
- Implement General-to-Specific Beam Search for the family dataset.
- Use `father` and `mother` facts as background knowledge.
- Learn `grandfather` rules from positive and negative examples.
- Show output and comparison between FOIL and Beam Search.

---

## 5. Dataset Description

The dataset contains **24 persons** from two families.

**Persons:**

```text
Christopher, Penelope, Arthur, Victoria, Andrew, Christine,
James, Jennifer, Colin, Charlotte, Margaret, Charles,
Roberto, Maria, Emilio, Lucia, Pierro, Francesca,
Marco, Angela, Alfonso, Sophia, Gina, Tomaso
```

**Background Predicate 1 — father(X,Y)** (12 facts):

```prolog
father(Christopher, Arthur).    father(Christopher, Victoria).
father(Andrew, James).          father(Andrew, Jennifer).
father(James, Colin).           father(James, Charlotte).
father(Roberto, Emilio).        father(Roberto, Lucia).
father(Pierro, Marco).          father(Pierro, Angela).
father(Marco, Alfonso).         father(Marco, Sophia).
```

**Background Predicate 2 — mother(X,Y)** (12 facts):

```prolog
mother(Penelope, Arthur).       mother(Penelope, Victoria).
mother(Christine, James).       mother(Christine, Jennifer).
mother(Victoria, Colin).        mother(Victoria, Charlotte).
mother(Maria, Emilio).          mother(Maria, Lucia).
mother(Francesca, Marco).       mother(Francesca, Angela).
mother(Lucia, Alfonso).         mother(Lucia, Sophia).
```

**Target Predicate — grandfather(X,Y)** (8 positive examples):

These are derived automatically: X is grandfather of Y if X is father of some Z, and Z is a parent of Y.

```prolog
grandfather(Christopher, Colin).      grandfather(Christopher, Charlotte).
grandfather(Andrew, Colin).           grandfather(Andrew, Charlotte).
grandfather(Roberto, Alfonso).        grandfather(Roberto, Sophia).
grandfather(Pierro, Alfonso).         grandfather(Pierro, Sophia).
```

**Negative examples** are generated using the closed-world assumption. All ordered pairs (X, Y) from 24 persons are generated. If a pair is not listed as a positive `grandfather(X,Y)` example, it is treated as negative. This gives **568 negative examples** (24 × 24 − 8).

---

## 6. FOIL Algorithm

FOIL means **First Order Inductive Learner**. It learns first-order logic rules from positive and negative examples. It starts with the most general rule and adds one literal at a time to make the rule more specific.

**Initial rule:**

```prolog
grandfather(X,Y) :- true.
```

**Simple pseudocode:**

```text
Start with an empty rule body
While the rule still covers negative examples:
    Generate candidate literals
    Calculate FOIL Gain for each literal
    Select the literal with the highest gain
    Add selected literal to the rule body
Remove positive examples covered by the rule
Repeat if uncovered positive examples remain
```

**Candidate literals used:**

```prolog
father(X,Z)    % X is father of some intermediate Z
father(Z,Y)    % Z is father of Y
mother(Z,Y)    % Z is mother of Y
father(X,Y)    % X is directly father of Y
mother(X,Z)    % X is mother of some Z
mother(X,Y)    % X is directly mother of Y
```

---

## 7. FOIL Gain

FOIL Gain is used to select the best literal at each step.

```text
Gain(L, R) = t × [ log2(p1 / (p1 + n1)) − log2(p0 / (p0 + n0)) ]
```

Where:

- `p0` = positive examples covered before adding literal
- `n0` = negative examples covered before adding literal
- `p1` = positive examples covered after adding literal
- `n1` = negative examples covered after adding literal
- `t` = number of positive examples still covered after adding literal

A useful literal keeps positive coverage high and reduces negative coverage.

---

## 8. Beam Search Algorithm

Beam Search is a **General-to-Specific** search method. It starts with the most general rule and creates more specific candidate rules by adding literals one at a time.

In this assignment:

- Beam width `k = 3`
- Max depth = 4
- Candidate rules are scored using **accuracy** and **coverage**
- The best 3 rules are kept at each level

**Scoring formulas:**

```text
Accuracy = p / (p + n)
Coverage = p / total_positives
```

Beam Search is less greedy than FOIL because it keeps multiple candidate rules during the search instead of committing to one literal at each step.

---

## 9. Implementation Details

**Project folder structure:**

```text
family_ilp/
|
|-- data/
|   `-- family_dataset.py
|
|-- src/
|   |-- family_loader.py
|   |-- family_foil.py
|   |-- family_beam_search.py
|   `-- main_family.py
|
|-- output/
|   |-- family_foil_output.txt
|   `-- family_beam_output.txt
```

**Important files:**

| File | Purpose |
| --- | --- |
| `data/family_dataset.py` | Stores all persons, father/mother facts, auto-computes grandfather examples |
| `src/family_loader.py` | Loads data, generates negative examples, defines candidate literals |
| `src/family_foil.py` | Implements simplified FOIL algorithm |
| `src/family_beam_search.py` | Implements General-to-Specific Beam Search |
| `src/main_family.py` | Runs the full project and writes output files |

---

## 10. Python Implementation Explanation

The dataset is stored in `data/family_dataset.py`. The `father` and `mother` facts are manually listed. The `GRANDFATHER` examples are automatically computed using a helper function:

```python
def _compute_grandfather(father_facts, mother_facts):
    """grandfather(X,Y) :- father(X,Z), (father(Z,Y) or mother(Z,Y))"""
    parent_facts = set(father_facts) | set(mother_facts)
    grandchildren_of = {}
    for x, z in father_facts:
        grandchildren_of.setdefault(x, set())
        for z2, y in parent_facts:
            if z2 == z:
                grandchildren_of[x].add(y)
    result = []
    for x, ys in grandchildren_of.items():
        for y in sorted(ys):
            result.append((x, y))
    return result
```

In `family_loader.py`, positive examples are loaded from `GRANDFATHER`. Negative examples are all (X, Y) pairs from 24 persons not in the positive set.

**FOIL learning process:**

**Rule 1 — Step 1:**
FOIL starts with `grandfather(X,Y) :- true.` covering all 8 positives and 568 negatives.
The best literal is `father(X,Z)` with FOIL Gain = **16.0000**.
After adding it: p1=8, n1=136.

**Rule 1 — Step 2:**
Now evaluates remaining literals. Best literal is `father(Z,Y)` with FOIL Gain = **16.6797**.
After adding: p1=4, n1=0. No negatives remain → rule is pure.

```prolog
Learned Rule 1: grandfather(X,Y) :- father(X,Z), father(Z,Y).
Covers: grandfather(Andrew, Colin), grandfather(Andrew, Charlotte),
        grandfather(Pierro, Alfonso), grandfather(Pierro, Sophia)
```

**Rule 2 — Step 1:**
Remaining 4 positives. Again best is `father(X,Z)` (gain = **8.1224**).

**Rule 2 — Step 2:**
Now `father(Z,Y)` covers 0 (already used by Rule 1 examples). Best is `mother(Z,Y)` with gain = **20.5171**.
After adding: p1=4, n1=0. Pure rule learned.

```prolog
Learned Rule 2: grandfather(X,Y) :- father(X,Z), mother(Z,Y).
Covers: grandfather(Christopher, Colin), grandfather(Christopher, Charlotte),
        grandfather(Roberto, Alfonso), grandfather(Roberto, Sophia)
```

All 8 positives covered. FOIL finishes.

---

## 11. FOIL Output (Actual)

```text
FOIL Learning — Target: grandfather(X,Y)
Positive examples : 8
Negative examples : 568

=== Learning Rule 1 ===
Start: grandfather(X,Y) :- true.

  Coverage: p0=8, n0=568
  Evaluating candidate literals:
    father(X,Z)           p1= 8, n1=136, gain=16.0000
    father(Z,Y)           p1= 8, n1=280, gain=8.0000
    mother(Z,Y)           p1= 8, n1=280, gain=8.0000
    father(X,Y)           p1= 0, n1= 12, gain=-inf
    mother(X,Z)           p1= 0, n1=144, gain=-inf
    mother(X,Y)           p1= 0, n1= 12, gain=-inf

  >> Selected: father(X,Z)  (gain=16.0000)
  Updated rule : grandfather(X,Y) :- father(X,Z).
  Coverage now : p1=8, n1=136

  Coverage: p0=8, n0=136
  Evaluating candidate literals:
    father(Z,Y)           p1= 4, n1= 0, gain=16.6797
    mother(Z,Y)           p1= 4, n1= 0, gain=16.6797
    ...

  >> Selected: father(Z,Y)  (gain=16.6797)
  Updated rule : grandfather(X,Y) :- father(X,Z), father(Z,Y).
  Coverage now : p1=4, n1=0

  No negatives covered — rule is pure. Done.
Learned Rule 1: grandfather(X,Y) :- father(X,Z), father(Z,Y).

=== Learning Rule 2 ===
  ...
  >> Selected: father(X,Z)  (gain=8.1224)
  >> Selected: mother(Z,Y)  (gain=20.5171)
Learned Rule 2: grandfather(X,Y) :- father(X,Z), mother(Z,Y).

Final Learned Rules:
  grandfather(X,Y) :- father(X,Z), father(Z,Y).
  grandfather(X,Y) :- father(X,Z), mother(Z,Y).
FOIL Finished.
```

---

## 12. Beam Search Output (Actual)

```text
Beam Search — Target: grandfather(X,Y)
Beam width=3, Max depth=4

=== Searching Rule 1 ===
Depth 1: 6 candidates
  Top 3 rules:
  1. grandfather(X,Y) :- father(X,Z).  acc=0.056, cov=1.000, p=8, n=136
  2. grandfather(X,Y) :- father(Z,Y).  acc=0.028, cov=1.000, p=8, n=280
  3. grandfather(X,Y) :- mother(Z,Y).  acc=0.028, cov=1.000, p=8, n=280

Depth 2: 15 candidates
  Top 3 rules:
  1. grandfather(X,Y) :- father(X,Z), father(Z,Y).  acc=1.000, cov=0.500, p=4, n=0
  2. grandfather(X,Y) :- father(X,Z), mother(Z,Y).  acc=1.000, cov=0.500, p=4, n=0
  3. grandfather(X,Y) :- father(Z,Y), father(X,Z).  acc=1.000, cov=0.500, p=4, n=0

Selected Rule 1: grandfather(X,Y) :- father(X,Z), father(Z,Y).

=== Searching Rule 2 ===
Depth 2: 15 candidates
  Top rule:
  1. grandfather(X,Y) :- father(X,Z), mother(Z,Y).  acc=1.000, cov=1.000, p=4, n=0

Selected Rule 2: grandfather(X,Y) :- father(X,Z), mother(Z,Y).

Final Rules:
  grandfather(X,Y) :- father(X,Z), father(Z,Y).
  grandfather(X,Y) :- father(X,Z), mother(Z,Y).
Beam Search Finished.
```

---

## 13. Result Analysis

Both algorithms learned the same two rules:

```prolog
grandfather(X,Y) :- father(X,Z), father(Z,Y).
grandfather(X,Y) :- father(X,Z), mother(Z,Y).
```

**Rule 1 explanation:**
X is grandfather of Y if X is the father of some Z, and Z is also the father of Y.
Example: `grandfather(Andrew, Colin)` because `father(Andrew, James)` and `father(James, Colin)`.

**Rule 2 explanation:**
X is grandfather of Y if X is the father of some Z, and Z is the mother of Y.
Example: `grandfather(Christopher, Colin)` because `father(Christopher, Victoria)` and `mother(Victoria, Colin)`.

These two rules together cover all 8 positive examples with zero false positives.

---

## 14. Comparison Between FOIL and Beam Search

| Criteria | FOIL | Beam Search |
| --- | --- | --- |
| Search type | General-to-specific | General-to-specific |
| Selection method | FOIL Gain (information gain) | Accuracy + Coverage score |
| Rules kept at each step | 1 (greedy) | k=3 (beam width) |
| Memory usage | Lower | Higher |
| Result | Same 2 rules learned | Same 2 rules learned |
| Speed | Faster | Slightly slower |
| Risk of missing best rule | Higher (greedy) | Lower (keeps alternatives) |

---

## 15. Why Prolog Is Not Used Here

The original graph assignment used Prolog to manually write rules and verify queries. In this assignment, the **goal is to automatically learn** those Prolog-style rules from data. So:

- **Python** acts as the learner — it discovers the rules using FOIL and Beam Search.
- **Prolog** would be the output — the learned rules can directly be written in Prolog.

If we wrote the rules in Prolog manually, that would not be machine learning. The point of ILP is to let the algorithm discover the rules on its own.

The equivalent Prolog representation of the learned rules would be:

```prolog
% Background knowledge
father(christopher, arthur).
father(christopher, victoria).
% ... (all 12 father facts)

mother(penelope, arthur).
mother(penelope, victoria).
% ... (all 12 mother facts)

% Learned rules
grandfather(X,Y) :- father(X,Z), father(Z,Y).
grandfather(X,Y) :- father(X,Z), mother(Z,Y).

% Sample queries:
% ?- grandfather(christopher, colin).   → true
% ?- grandfather(andrew, charlotte).    → true
% ?- grandfather(penelope, colin).      → false
```

---

## 16. Limitations

- This is a simplified ILP implementation.
- Only `grandfather` is learned as the target predicate. Other relations like `uncle`, `aunt`, `grandmother` could also be learned with different candidate literals.
- The closed-world assumption generates a large number of negative examples (568) compared to positives (8), which creates a class imbalance.
- FOIL is greedy — it may miss a globally better rule if a locally weaker literal leads to a better overall rule.
- Recursive rules (like in the graph path problem) are not explored here.

---

## 17. Conclusion

In this assignment, FOIL and Beam Search were applied to a family relations dataset to learn the `grandfather` predicate from `father` and `mother` background facts. Both algorithms successfully learned two correct and complete rules:

```prolog
grandfather(X,Y) :- father(X,Z), father(Z,Y).
grandfather(X,Y) :- father(X,Z), mother(Z,Y).
```

Rule 1 covers the case where the grandfather's son is the father of the grandchild. Rule 2 covers the case where the grandfather's daughter is the mother of the grandchild. Together they cover all 8 positive examples with perfect accuracy.

This shows that ILP can automatically discover meaningful relational rules from structured data without manual rule writing.

---

## 18. References

- Class lecture slide: Learning First Order Logic / Learning Rules
- FOIL Algorithm — Quinlan, J.R. (1990)
- Machine Learning rule learning and ILP concepts
- SWI-Prolog documentation
