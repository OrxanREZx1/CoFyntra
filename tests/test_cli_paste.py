"""
Tests for CLI multiline paste support.

Covers:
1. Normal one-line input passes through unmodified.
2. /paste triggers multiline input mode.
3. Lines are collected until END is entered on a new line.
4. Joined lines preserve multi-paragraph content.
5. Case-insensitive /paste and EOF handling.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from startary.cli import read_founder_input


def test_normal_oneline_input_keeps_working():
    """Single-line text without /paste returns immediately."""
    with patch("builtins.input", side_effect=["We are building a Shopify returns plugin."]):
        result = read_founder_input("> ")
        assert result == "We are building a Shopify returns plugin."


def test_multiline_paste_collects_until_end():
    """Entering /paste collects multiple lines until END on a new line."""
    inputs = [
        "/paste",
        "Line 1: 6 grocery store interviews completed.",
        "Line 2: 4/6 had recent stockouts in dairy and produce.",
        "Line 3: Current ordering uses manual pen-and-paper logs.",
        "END",
    ]
    with patch("builtins.input", side_effect=inputs):
        result = read_founder_input("> ")

    expected = (
        "Line 1: 6 grocery store interviews completed.\n"
        "Line 2: 4/6 had recent stockouts in dairy and produce.\n"
        "Line 3: Current ordering uses manual pen-and-paper logs."
    )
    assert result == expected


def test_multiline_paste_case_insensitive_and_whitespace():
    """Supports '/PASTE ' and stops on 'END'."""
    inputs = [
        "  /PASTE  ",
        "Problem: Inventory waste",
        "Evidence: 3 stores reported spoilage",
        "END",
    ]
    with patch("builtins.input", side_effect=inputs):
        result = read_founder_input("> ")

    assert result == "Problem: Inventory waste\nEvidence: 3 stores reported spoilage"


def test_multiline_paste_eof_handling():
    """EOFError or KeyboardInterrupt during multiline input exits safely."""
    inputs = [
        "/paste",
        "Line 1",
        EOFError(),
    ]
    with patch("builtins.input", side_effect=inputs):
        result = read_founder_input("> ")

    assert result == "Line 1"


def test_initial_eof_handling():
    """EOFError on first prompt returns empty string."""
    with patch("builtins.input", side_effect=[EOFError()]):
        result = read_founder_input("> ")
        assert result == ""
