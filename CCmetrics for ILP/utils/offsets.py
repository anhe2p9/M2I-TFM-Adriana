# Import the required libraries
from typing import Tuple


# Get if one extraction (e1) is contained in another (e2)
# Return True when e1 is contained in e2, false otherwise
def is_contained(e1: Tuple[int, int], e2: Tuple[int, int]) -> bool:
    return (e1[0] >= e2[0] and e1[1] <= e2[1])


# Get if two extractions are not related (contained) each other
# Return True when the two extractions disjoint and false when e2 is related (contained) in e1
def disjoint(e1: Tuple[int, int], e2: Tuple[int, int]) -> bool:
    return e2[0] > e1[1] or e1[0] > e2[1]


# Get if two extractions are in conflict
# Return True when e1 and e2 overlaps (they are in conflict), false otherwise
def is_in_conflict(e1: Tuple[int, int], e2: Tuple[int, int]) -> bool:
    return not (disjoint(e1, e2) or is_contained(e1, e2) or is_contained(e2, e1))
