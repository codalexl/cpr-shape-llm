"""The permutation test used in the results chapter, on labelings with a known p."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from dial_inference import exact_permutation


def test_complete_separation_of_two_and_two():
    diff, one, two = exact_permutation([1, 1], [0, 0])
    # six relabellings, two of them as extreme as the observed one
    assert diff == 1
    assert one == 1 / 6
    assert abs(two - 2 / 6) < 1e-12


def test_five_against_five_separation_is_two_over_252():
    _, _, two = exact_permutation([1] * 5, [0] * 5)
    assert abs(two - 2 / 252) < 1e-12
