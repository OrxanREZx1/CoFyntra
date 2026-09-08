"""
Tests: canonical startup taxonomy.

Verifies that the taxonomy is valid, centralised, and contains exactly
the ten areas specified in the architecture contract.
"""

import pytest
from startary.domain.taxonomy import StartupArea


EXPECTED_AREAS = {
    "Problem",
    "Target User",
    "Pain / Urgency",
    "Existing Alternatives",
    "Value Proposition",
    "Demand Signal",
    "Acquisition / Customer Access",
    "Business Model / Revenue",
    "Solution / MVP",
    "Founder Constraints",
}


def test_taxonomy_has_exactly_ten_areas():
    """The taxonomy must contain exactly ten canonical areas."""
    assert len(StartupArea) == 10


def test_taxonomy_values_match_spec():
    """All specified area names must be present."""
    actual = {area.value for area in StartupArea}
    assert actual == EXPECTED_AREAS


def test_taxonomy_no_duplicates():
    """Each area name must appear exactly once."""
    values = [area.value for area in StartupArea]
    assert len(values) == len(set(values))


def test_taxonomy_is_str_enum():
    """StartupArea members must be usable as plain strings."""
    area = StartupArea.PROBLEM
    assert isinstance(area, str)
    assert area == "Problem"


def test_taxonomy_iterable():
    """All areas can be iterated without error."""
    areas = list(StartupArea)
    assert len(areas) == 10


def test_taxonomy_dict_key_usability():
    """StartupArea members must be usable as dictionary keys."""
    d = {area: f"description of {area}" for area in StartupArea}
    assert len(d) == 10
    assert d[StartupArea.PROBLEM] == "description of Problem"

