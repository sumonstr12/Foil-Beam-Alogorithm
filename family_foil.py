"""FOIL algorithm adapted for the family dataset."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from family_loader import (
    candidate_literals, covered_examples, dataset_summary,
    example_pair, format_rule, load_family_dataset,
)


def _log_fraction(p, n):
    if p == 0 or p + n == 0:
        return None
    return math.log2(p / (p + n))


def foil_gain(p0, n0, p1, n1, t):
    old = _log_fraction(p0, n0)
    new = _log_fraction(p1, n1)
    if old is None or new is None:
        return float("-inf")
    return t * (new - old)


def _gain_text(v):
    return "-inf" if v == float("-inf") else f"{v:.4f}"


def learn_foil(positives, negatives, facts, literals):
    uncovered = positives[:]
    learned_rules = []
    output = [
        "FOIL Learning — Target: grandfather(X,Y)",
        f"Positive examples : {len(positives)}",
        f"Negative examples : {len(negatives)}",
        "",
    ]

    rule_number = 1

    while uncovered:
        rule = []
        output.append(f"=== Learning Rule {rule_number} ===")
        output.append(f"Start: {format_rule(rule)}")
        output.append("")

        while True:
            cur_pos = covered_examples(rule, uncovered, facts)
            cur_neg = covered_examples(rule, negatives, facts)
            p0, n0 = len(cur_pos), len(cur_neg)

            if n0 == 0:
                output.append("  No negatives covered — rule is pure. Done.")
                break

            best_lit, best_gain = None, float("-inf")
            best_pos, best_neg = [], []

            output.append(f"  Coverage: p0={p0}, n0={n0}")
            output.append("  Evaluating candidate literals:")

            for lit in literals:
                if lit in rule:
                    continue
                new_rule = rule + [lit]
                np = covered_examples(new_rule, uncovered, facts)
                nn = covered_examples(new_rule, negatives, facts)
                p1, n1 = len(np), len(nn)
                gain = foil_gain(p0, n0, p1, n1, p1)
                output.append(f"    {str(lit):20s}  p1={p1:2d}, n1={n1:2d}, gain={_gain_text(gain)}")

                if gain > best_gain:
                    best_lit, best_gain = lit, gain
                    best_pos, best_neg = np, nn

            if best_lit is None or best_gain <= 0:
                output.append("  No useful literal. Stopping rule.")
                break

            rule.append(best_lit)
            output.append(f"\n  >> Selected: {best_lit}  (gain={_gain_text(best_gain)})")
            output.append(f"  Updated rule : {format_rule(rule)}")
            output.append(f"  Coverage now : p1={len(best_pos)}, n1={len(best_neg)}")
            output.append("")

        covered = covered_examples(rule, uncovered, facts)
        if not covered:
            output.append("Rule covers nothing new. Stopping.")
            break

        covered_pairs = {example_pair(e) for e in covered}
        uncovered = [e for e in uncovered if example_pair(e) not in covered_pairs]

        learned_rules.append(rule)
        output.append(f"Learned Rule {rule_number}: {format_rule(rule)}")
        output.append(f"  Covered {len(covered)} positives | Remaining: {len(uncovered)}")
        output.append("")
        rule_number += 1

    output.append("=" * 40)
    output.append("Final Learned Rules:")
    for r in learned_rules:
        output.append(f"  {format_rule(r)}")
    output.append("FOIL Finished.")
    return learned_rules, "\n".join(output)


if __name__ == "__main__":
    persons, facts, positives, negatives = load_family_dataset()
    literals = candidate_literals()
    summary = dataset_summary(persons, facts, positives, negatives)
    _, text = learn_foil(positives, negatives, facts, literals)
    print(summary)
    print()
    print(text)
