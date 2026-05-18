"""Main runner — family dataset ILP."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from family_loader import candidate_literals, dataset_summary, load_family_dataset
from family_foil import learn_foil
from family_beam_search import learn_with_beam_search

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

persons, facts, positives, negatives = load_family_dataset()
literals = candidate_literals()

summary = dataset_summary(persons, facts, positives, negatives)
_, foil_text = learn_foil(positives, negatives, facts, literals)
_, beam_text = learn_with_beam_search(positives, negatives, facts, literals)

(output_dir / "family_foil_output.txt").write_text(summary + "\n\n" + foil_text, encoding="utf-8")
(output_dir / "family_beam_output.txt").write_text(summary + "\n\n" + beam_text, encoding="utf-8")

print(summary)
print()
print(foil_text)
print()
print("=" * 60)
print()
print(beam_text)
