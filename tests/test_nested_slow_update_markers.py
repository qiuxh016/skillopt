"""Regression coverage for nested protected Skill regions."""
from __future__ import annotations

import unittest

from skillopt.marker_utils import find_marked_region, is_within_marked_region
from skillopt.evaluation.gate import compute_semantic_density


START = "<!-- SLOW_UPDATE_START -->"
END = "<!-- SLOW_UPDATE_END -->"


class NestedSlowUpdateMarkerTests(unittest.TestCase):
    def test_nested_markers_match_the_outer_end(self) -> None:
        skill = f"body\n{START}\n{START}\ninner\n{END}\ntail\n{END}\nafter"
        bounds = find_marked_region(skill, START, END)
        self.assertIsNotNone(bounds)
        assert bounds is not None
        self.assertEqual(skill[bounds[1]:], "\nafter")

    def test_text_after_inner_end_remains_protected(self) -> None:
        skill = f"{START}\n{START}\ninside\n{END}\nTAIL_RULE\n{END}"
        self.assertTrue(is_within_marked_region(skill, "TAIL_RULE", START, END))

    def test_metadata_inside_nested_region_is_excluded_from_density(self) -> None:
        skill = f"ordinary words\n{START}\n{START}\nMUST ALWAYS VERIFY\n{END}\nTAIL_RULE\n{END}"
        self.assertEqual(compute_semantic_density(skill), 0.0)

    def test_multiple_separate_regions_are_matched_independently(self) -> None:
        skill = f"{START}first{END} middle {START}second{END}"
        first = find_marked_region(skill, START, END)
        self.assertIsNotNone(first)
        assert first is not None
        second = find_marked_region(skill, START, END, first[1])
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(skill[first[0]:first[1]], f"{START}first{END}")
        self.assertEqual(skill[second[0]:second[1]], f"{START}second{END}")

    def test_unclosed_region_is_not_reported_as_balanced(self) -> None:
        self.assertIsNone(find_marked_region(f"{START}unfinished", START, END))


if __name__ == "__main__":
    unittest.main()
