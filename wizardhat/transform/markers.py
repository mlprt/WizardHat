
from wizardhat import utils

import numpy as np


def rarest(markers):
    """Return a function that is true for marker values equal to the rarest."""
    unique, counts = np.unique(markers, return_counts=True)
    a = unique[np.argmin(counts)]
    def rule(n):
        return n == a
    return rule

def notzero(markers):
    """Return a function that is true for any non-zero marker value."""
    def rule(n):
        return n != 0
    return rule

def anybut(markers):
    """Return a function that admits any marker but the most common in `markers`."""
    unique, counts = np.unique(markers, return_counts=True)
    a = unique[np.argmax(counts)]
    def rule(n):
        return n != a
    return rule

def steps(markers):
    """Return a function that returns the previous value in a linear sequence.

    Assumes markers do not start negative and increase, with a value at 0.
    """
    marker_diffs = np.diff(markers[markers > 0])
    if len(set(marker_diffs)) == 1:
        b = marker_diffs[0]
    else:
        print("Given markers are not a linearly increasing sequence")
    def rule(n):
        return n - b
    return rule

MARKER_RULES = dict(sparsest=sparsest, notzero=notzero, anybut=anybut,
                    steps=steps)
