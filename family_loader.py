"""Family dataset loader — mirrors dataset.py but for family relations."""

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


@dataclass(frozen=True)
class Literal:
    predicate: str
    arguments: tuple

    def __str__(self):
        return f"{self.predicate}({','.join(self.arguments)})"


def _load_raw():
    data_path = Path(__file__).resolve().parent / "family_dataset.py"
    spec = spec_from_file_location("family_dataset", data_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_family_dataset(target="grandfather"):
    raw = _load_raw()
    persons = list(raw.PERSONS)

    facts = {
        "father": set(raw.FATHER),
        "mother": set(raw.MOTHER),
    }

    positive_pairs = set(raw.GRANDFATHER)
    positives = [{"X": x, "Y": y} for x, y in sorted(positive_pairs)]

    negatives = []
    for x in persons:
        for y in persons:
            if (x, y) not in positive_pairs:
                negatives.append({"X": x, "Y": y})

    return persons, facts, positives, negatives


def candidate_literals():
    """Candidate literals for learning grandfather(X,Y)."""
    return [
        Literal("father", ("X", "Z")),   # X is father of some Z
        Literal("father", ("Z", "Y")),   # Z is father of Y
        Literal("mother", ("Z", "Y")),   # Z is mother of Y
        Literal("father", ("X", "Y")),   # X is directly father of Y
        Literal("mother", ("X", "Z")),   # X is mother of some Z (grandmother path)
        Literal("mother", ("X", "Y")),   # X is directly mother of Y
    ]


def apply_literal(literal, bindings, facts):
    new_bindings = []
    fact_set = facts.get(literal.predicate, set())

    for binding in bindings:
        for fact in fact_set:
            if len(fact) != len(literal.arguments):
                continue
            possible = dict(binding)
            ok = True
            for arg, value in zip(literal.arguments, fact):
                if arg in possible and possible[arg] != value:
                    ok = False
                    break
                possible[arg] = value
            if ok:
                new_bindings.append(possible)

    return new_bindings


def rule_bindings(rule, example, facts):
    bindings = [dict(example)]
    for literal in rule:
        bindings = apply_literal(literal, bindings, facts)
        if not bindings:
            break
    return bindings


def rule_covers(rule, example, facts):
    return bool(rule_bindings(rule, example, facts))


def covered_examples(rule, examples, facts):
    return [e for e in examples if rule_covers(rule, e, facts)]


def format_rule(rule, target="grandfather"):
    if not rule:
        return f"{target}(X,Y) :- true."
    body = ", ".join(str(lit) for lit in rule)
    return f"{target}(X,Y) :- {body}."


def example_pair(example):
    return example["X"], example["Y"]


def dataset_summary(persons, facts, positives, negatives):
    father_list = sorted(facts["father"])
    mother_list = sorted(facts["mother"])
    lines = [
        "Family Dataset — Target: grandfather(X,Y)",
        "------------------------------------------",
        f"Persons ({len(persons)}): {', '.join(persons[:8])} ...",
        "",
        f"father facts : {len(father_list)}",
        f"mother facts : {len(mother_list)}",
        "",
        f"Positive grandfather examples : {len(positives)}",
        f"Negative examples             : {len(negatives)}",
        "(Closed-world assumption: unlisted pairs are negative)",
        "",
        "Positive examples:",
    ]
    for p in positives:
        lines.append(f"  grandfather({p['X']}, {p['Y']})")
    return "\n".join(lines)
