
from wizardhat import utils

import numpy as np


def sparsest(markers):
    """"""
    unique, counts = np.unique(markers, return_counts=True)
    def rule(n):
        return unique[np.argmin(counts)]
    return rule

def notzero(markers):
    def rule(n):
        return n != 0
    return rule

def anybut(markers):
    """Return a function that admits any marker but the most common in `markers`."""
    unique, counts = np.unique(markers, return_counts=True)
    a = unique[np.argmax(counts)]
    def rule(n):
        return n != a

def steps(markers):
    marker_diffs = np.diff(markers[markers > 0])
    if np.std(marker_diffs) > :
        return

MARKER_RULES = dict(sparsest=sparsest, notzero=notzero, steps=steps)
