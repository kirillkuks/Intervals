from __future__ import annotations
from typing import List


class Interval:
    @staticmethod
    def min_max_union(intervals: List[Interval]) -> Interval:
        union_interval = intervals[0]

        for interval in intervals:
            union_interval = Interval(
                min(union_interval.left, interval.left),
                max(union_interval.right, interval.right)
            )

        return union_interval
    
    @staticmethod
    def min_max_intersection(intervals: List[Interval]) -> Interval:
        intersection_interval = intervals[0]

        for interval in intervals:
            intersection_interval = Interval(
                max(intersection_interval.left, interval.left),
                min(intersection_interval.right, interval.right)
            )

        return intersection_interval
    
    @staticmethod
    def jaccard_index(intervals: List[Interval]) -> float:
        return Interval.min_max_intersection(intervals).wid() / Interval.min_max_union(intervals).wid() * 0.5 + 0.5

    
    @staticmethod
    def scale_intervals(intervals: List[Interval], multiplier: float) -> List[Interval]:
        return [interval.scale(multiplier) for interval in intervals]
    
    @staticmethod
    def expand_intervals(intervals: List[Interval], eps: float) -> List[Interval]:
        return [interval.expand(eps) for interval in intervals]
    
    @staticmethod
    def combine_intervals(intervals1 : List[Interval], intervals2: List[Interval]) -> List[Interval]:
        return [j for i in [intervals1, intervals2] for j in i]


    def __init__(self, x: float, y: float, force_right: bool = False) -> None:
        self.left =  min(x, y) if force_right else x
        self.right = max(x, y) if force_right else y

    def wid(self) -> float:
        return self.right - self.left
    
    def rad(self) -> float:
        return self.wid() * 0.5
    
    def mid(self) -> float:
         return (self.left + self.right) * 0.5
    
    def pro(self) -> Interval:
        return Interval(self.left, self.right, True)
    
    def scale(self, multiplier: float) -> Interval:
        return Interval(self.left * multiplier, self.right * multiplier, True)
    
    def expand(self, eps: float) -> Interval:
        return Interval(self.left - eps, self.right + eps)

    def to_str(self) -> str:
        return f'[{self.left}, {self.right}]'
