"""Beam Search adapted for the family dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from family_loader import (
    candidate_literals, covered_examples, dataset_summary,
    example_pair, format_rule, load_family_dataset,
)


def evaluate_rule(rule, positives, negatives, facts):
    pos = covered_examples(rule, positives, facts)
    neg = covered_examples(rule, negatives, facts)
    p, n = len(pos), len(neg)
    total = p + n
    return {
        "p": p, "n": n,
        "accuracy": p / total if total else 0.0,
        "coverage": p / len(positives) if positives else 0.0,
    }


def _rule_key(rule):
    return tuple(str(lit) for lit in rule)


def _sort_key(item):
    _, s = item
    return (s["accuracy"], s["coverage"], s["p"], -s["n"])


def beam_search_one_rule(positives, negatives, facts, literals, beam_width=3, max_depth=4):
    beam = [[]]
    best_rule = []
    best_score = evaluate_rule([], positives, negatives, facts)
    seen = {()}
    lines = []

    for depth in range(1, max_depth + 1):
        candidates = []
        for rule in beam:
            for lit in literals:
                if lit in rule:
                    continue
                new_rule = rule + [lit]
                key = _rule_key(new_rule)
                if key in seen:
                    continue
                seen.add(key)
                score = evaluate_rule(new_rule, positives, negatives, facts)
                candidates.append((new_rule, score))

        if not candidates:
            break

        candidates.sort(key=_sort_key, reverse=True)
        beam = [r for r, _ in candidates[:beam_width]]

        lines.append(f"Depth {depth}: {len(candidates)} candidates")
        lines.append(f"  Top {min(beam_width, len(candidates))} rules:")
        for i, (rule, score) in enumerate(candidates[:beam_width], 1):
            lines.append(
                f"  {i}. {format_rule(rule)}"
                f"  acc={score['accuracy']:.3f}, cov={score['coverage']:.3f},"
                f" p={score['p']}, n={score['n']}"
            )
            if _sort_key((rule, score)) > _sort_key((best_rule, best_score)):
                best_rule, best_score = rule, score
        lines.append("")

        if best_score["accuracy"] == 1.0 and best_score["coverage"] == 1.0:
            break

    return best_rule, best_score, lines


def learn_with_beam_search(positives, negatives, facts, literals, beam_width=3, max_depth=4):
    uncovered = positives[:]
    learned_rules = []
    output = [
        "Beam Search — Target: grandfather(X,Y)",
        f"Beam width={beam_width}, Max depth={max_depth}",
        "",
    ]

    rule_number = 1
    while uncovered:
        output.append(f"=== Searching Rule {rule_number} ===")
        rule, score, detail = beam_search_one_rule(
            uncovered, negatives, facts, literals,
            beam_width=beam_width, max_depth=max_depth,
        )
        output.extend(detail)

        covered = covered_examples(rule, uncovered, facts)
        if not covered:
            output.append("No useful rule found.")
            break

        covered_pairs = {example_pair(e) for e in covered}
        uncovered = [e for e in uncovered if example_pair(e) not in covered_pairs]

        learned_rules.append(rule)
        output.append(f"Selected Rule {rule_number}: {format_rule(rule)}")
        output.append(
            f"  acc={score['accuracy']:.3f}, cov={score['coverage']:.3f},"
            f" p={score['p']}, n={score['n']}"
        )
        output.append(f"  Remaining positives: {len(uncovered)}")
        output.append("")
        rule_number += 1

    output.append("=" * 40)
    output.append("Final Rules:")
    for r in learned_rules:
        output.append(f"  {format_rule(r)}")
    output.append("Beam Search Finished.")
    return learned_rules, "\n".join(output)


if __name__ == "__main__":
    persons, facts, positives, negatives = load_family_dataset()
    literals = candidate_literals()
    summary = dataset_summary(persons, facts, positives, negatives)
    _, text = learn_with_beam_search(positives, negatives, facts, literals)
    print(summary)
    print()
    print(text)
