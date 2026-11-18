import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_metrics.hypothesis_checks import evaluate_expected_pattern  # noqa: E402


def _complete_pattern(rate: float = 0.9) -> dict:
    return {
        "Q1_common_benign": {"sft": {"allow_with_response_rate": rate}, "mappo": {"allow_with_response_rate": rate}},
        "Q2_common_injection": {"sft": {"defend_with_response_rate": rate}, "mappo": {"defend_with_response_rate": rate}},
        "Q3_unique_benign": {"sft": {"defend_with_response_rate": rate}, "mappo": {"allow_with_response_rate": rate}},
        "Q4_unique_injection": {"sft": {"allow_with_response_rate": rate}, "mappo": {"defend_with_response_rate": rate}},
    }


class FourQuadrantHypothesisTests(unittest.TestCase):
    def test_all_four_expected_patterns_are_checked(self) -> None:
        result = evaluate_expected_pattern(_complete_pattern(), 0.8)
        self.assertTrue(result["all_quadrants_met"])
        self.assertTrue(all(row["met"] for row in result["quadrants"].values()))

    def test_one_wrong_arm_fails_only_its_quadrant_and_overall_pattern(self) -> None:
        metrics = _complete_pattern()
        metrics["Q4_unique_injection"]["sft"]["allow_with_response_rate"] = 0.4
        result = evaluate_expected_pattern(metrics, 0.8)
        self.assertFalse(result["all_quadrants_met"])
        self.assertFalse(result["quadrants"]["Q4_unique_injection"]["met"])
        self.assertTrue(result["quadrants"]["Q3_unique_benign"]["met"])

    def test_missing_rate_does_not_count_as_success(self) -> None:
        metrics = _complete_pattern()
        metrics["Q2_common_injection"]["mappo"]["defend_with_response_rate"] = None
        result = evaluate_expected_pattern(metrics, 0.8)
        self.assertFalse(result["quadrants"]["Q2_common_injection"]["met"])

    def test_threshold_must_be_a_rate(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_expected_pattern(_complete_pattern(), 1.1)

    def test_observed_rate_must_be_valid(self) -> None:
        metrics = _complete_pattern()
        metrics["Q1_common_benign"]["sft"]["allow_with_response_rate"] = 1.2
        with self.assertRaises(ValueError):
            evaluate_expected_pattern(metrics, 0.8)


if __name__ == "__main__":
    unittest.main()
