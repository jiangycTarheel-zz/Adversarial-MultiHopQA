"""Data structures for dealing with intervals."""
import argparse
import sys

OPTS = None

class Interval(object):
  """Represents a half-open interval."""
  def __init__(self, start, end, value=None):
    self.start = start
    self.end = end
    self.value = value

  def contains_pt(self, pt):
    return pt >= self.start and pt < self.end

  def contains(self, other):
    return self.start <= other.start and self.end >= other.end

  def overlaps(self, other, closed_boundaries=False):
    if closed_boundaries:
      return not (self.start > other.end or self.end < other.start)
    return not (self.start >= other.end or self.end <= other.start)

  def overlap_len(self, other):
    start = max(self.start, other.start)
    end = min(self.end, other.end)
    if start > end: return 0
    return end - start

  def length(self):
    return self.end - self.start

  def __key(self):
    return (self.start, self.end)

  def __eq__(self, other):
    return self.__key() == other.__key()

  def __hash__(self):
    return hash(self.__key())

  def __lt__(self, other):
    return self.__key() < other.__key()

  def __str__(self):
    return str((self.start, self.end))

class IntervalSet(object):
  """Represents a monotincally growing set of half-open intervals."""
  def __init__(self):
    self.intervals = []

  @classmethod
  def from_list(cls, interval_list):
    ret = cls()
    ret.intervals = sorted(interval_list)
    return ret

  def _extend_match(self, index, interval, closed_boundaries=False):
    # Linear search forward and backward
    start = index
    while start-1 >= 0 and self.intervals[start-1].overlaps(interval):
      start -= 1
    end = index
    while end+1 < len(self.intervals) and self.intervals[end+1].overlaps(interval):
      end += 1
    return (start, end + 1)  # Return half-open interval

  def search(self, interval, closed_boundaries=False):
    """Search for all overlapping intervals.
    
    Returns (Found/not Found, start_ind, end_ind)
    """
    lo = 0
    hi = len(self.intervals)
    while hi - lo > 4:
      mid = (lo + hi) / 2
      x = self.intervals[mid]
      if x.overlaps(interval, closed_boundaries=closed_boundaries):
        return (True,) + self._extend_match(mid, interval, closed_boundaries=closed_boundaries)
      elif x.start >= interval.end:
        hi = mid
      elif x.end <= interval.start:
        lo = mid
    for i in range(lo, hi):
      x = self.intervals[i]
      if x.overlaps(interval):
        return (True,) + self._extend_match(i, interval, closed_boundaries=closed_boundaries)
      elif x.start >= interval.end:
        # We should insert at this index
        return (False, i, i)
    return (False, hi, hi)

  def contains(self, interval):
    found, start_ind, end_ind = self.search(interval)
    if not found: return False
    if start_ind - end_ind != 1: return False
    return self.intervals[start_ind].contains(interval)

  def overlaps(self, interval, closed_boundaries=False):
    return self.search(interval, closed_boundaries=closed_boundaries)[0]

  def add(self, interval):
    found, start_ind, end_ind = self.search(interval, closed_boundaries=True)
    if found:
      if start_ind - end_ind == 1 and self.intervals[start_ind].contains(interval):
        return
      before_elems = self.intervals[:start_ind]
      after_elems = self.intervals[end_ind:]
      new_start = min((interval.start, self.intervals[start_ind].start))
      new_end = max((interval.end, self.intervals[end_ind - 1].end))
      interval = Interval(new_start, new_end)
      self.intervals = before_elems + [interval] + after_elems
    else:
      self.intervals.insert(start_ind, interval)
    # Check if sorted
#    if any(self.intervals[i] > self.intervals[i+1] for i in range(len(self.intervals) - 1)):
#      for x in self.intervals:
#        print x
#      raise ValueError('Not sorted!')

#    if self.contains(interval): return
#    new_start = interval.start
#    new_end = interval.end
#    to_remove = set()
#    for x in self.intervals:
#      if interval.contains(x):
#        to_remove.add(x)
#      else:
#        if x.end >= interval.start and x.start < new_start:
#          new_start = x.start
#          to_remove.add(x)
#        if x.start <= interval.end and x.end > new_end:
#          new_end = x.end
#          to_remove.add(x)
#    new_interval = Interval(new_start, new_end)
#    for x in to_remove:
#      self.intervals.remove(x)
#    self.intervals.append(new_interval)
#    self.intervals.sort()   # TODO: binary search?

  def complement(self, interval, min_size=0):
    """Return the complement of current set within the given interval."""
    cur_start = interval.start
    new_intervals = []
    for x in self.intervals:  # Relies on self.intervals being sorted
      if x.end < interval.start: continue
      if x.start > interval.end: continue
      if x.start < interval.start:
        cur_start = x.end
      else:
        cur_end = x.start
        new_intervals.append((Interval(cur_start, cur_end)))
        cur_start = x.end
    if cur_start < interval.end:
      new_intervals.append(Interval(cur_start, interval.end))
    return IntervalSet.from_list(new_intervals)
