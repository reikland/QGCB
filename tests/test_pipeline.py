import unittest

from qgcb.models import ProtoQuestion
from qgcb.pipeline import (
    enforce_type_distribution,
    normalize_question_entry,
    serialize_questions_to_csv,
)


class TypeDistributionTests(unittest.TestCase):
    def _make(self, q_type: str) -> ProtoQuestion:
        return ProtoQuestion(title=f"{q_type} title", question="Q?", type=q_type)

    def test_enforce_type_distribution_matches_target(self):
        questions = (
            [self._make("binary") for _ in range(5)]
            + [self._make("numeric") for _ in range(3)]
            + [self._make("multiple_choice") for _ in range(2)]
        )
        res = enforce_type_distribution(questions)
        self.assertEqual(res["counts"], {"binary": 5, "numeric": 3, "multiple_choice": 2, "unknown": 0})

    def test_enforce_type_distribution_raises_on_drift(self):
        questions = [self._make("binary") for _ in range(9)] + [self._make("numeric")]
        with self.assertRaises(ValueError):
            enforce_type_distribution(questions)


class CsvNormalizationTests(unittest.TestCase):
    def test_normalize_and_serialize_questions(self):
        row = {
            "id": "q1",
            "title": "Sample",
            "question": "What is the outcome?",
            "type": "Multiple choice",
            "options": ["A", "B", "C"],
            "category": "economics",
            "question_weight": 1,
            "role": "CORE",
            "group_variable": "sector",
            "range_min": None,
            "range_max": None,
            "candidate_source": "Reuters",
            "tags": ["finance", "markets"],
            "rating": "Publishable",
            "rating_rationale": "Fits guidelines",
            "resolution_card": "Resolve when reported",
        }

        normalized = normalize_question_entry(row)
        self.assertEqual(normalized["type"], "multiple_choice")
        self.assertEqual(normalized["options"], "A|B|C")
        self.assertEqual(normalized["tags"], "finance, markets")

        csv_data = serialize_questions_to_csv([row], extra_fields=["resolution_card"])
        self.assertIn("multiple_choice", csv_data)
        self.assertIn("resolution_card", csv_data.split("\n")[0])


if __name__ == "__main__":
    unittest.main()
