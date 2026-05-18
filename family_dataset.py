"""Family relations dataset."""

PERSONS = [
    "Christopher", "Penelope", "Arthur", "Victoria",
    "Andrew", "Christine", "James", "Jennifer",
    "Colin", "Charlotte", "Margaret", "Charles",
    "Roberto", "Maria", "Emilio", "Lucia",
    "Pierro", "Francesca", "Marco", "Angela",
    "Alfonso", "Sophia", "Gina", "Tomaso",
]

FATHER = [
    ("Christopher", "Arthur"), ("Christopher", "Victoria"),
    ("Andrew", "James"), ("Andrew", "Jennifer"),
    ("James", "Colin"), ("James", "Charlotte"),
    ("Roberto", "Emilio"), ("Roberto", "Lucia"),
    ("Pierro", "Marco"), ("Pierro", "Angela"),
    ("Marco", "Alfonso"), ("Marco", "Sophia"),
]

MOTHER = [
    ("Penelope", "Arthur"), ("Penelope", "Victoria"),
    ("Christine", "James"), ("Christine", "Jennifer"),
    ("Victoria", "Colin"), ("Victoria", "Charlotte"),
    ("Maria", "Emilio"), ("Maria", "Lucia"),
    ("Francesca", "Marco"), ("Francesca", "Angela"),
    ("Lucia", "Alfonso"), ("Lucia", "Sophia"),
]


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


GRANDFATHER = _compute_grandfather(FATHER, MOTHER)
